"""Rewrite the image paths in the training data to point at your copy of the screenshots.

The training data stores paths with a placeholder root, such as
    /path/to/screenshots/images/Aviation/Aviation_7.png
Everything from the `images/` (or `injected_images/`) folder onwards is kept,
and the part before it is replaced with the prefix you provide:
    <new_prefix>/images/Aviation/Aviation_7.png

<new_prefix> is the directory that contains the `images/` and `injected_images/`
folders. It can be anywhere, e.g. a separate data disk; it does not have to be
inside this repository.

Supported files:
    *.json     LLaMA-Factory SFT data  (field: "images": [path, ...])
    *.parquet  verl RL data            (field: "images": [{"image": path, ...}, ...]), requires pandas + pyarrow

Usage:
    python replace_image_paths.py --new_prefix /path/to/screenshots
    python replace_image_paths.py --new_prefix /path/to/screenshots --files sft/LLaMA-Factory/data/detection_sft_data.json
"""

import argparse
import json
import os
import re
import shutil

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = [
    os.path.join(TRAIN_DIR, "sft/LLaMA-Factory/data/detection_sft_data.json"),
    os.path.join(TRAIN_DIR, "rl/verl/data/train.parquet"),
    os.path.join(TRAIN_DIR, "rl/verl/data/test.parquet"),
]
# Splits "<old_prefix>/<images|injected_images>/<rest>" at the image folder.
IMAGE_DIR_PATTERN = re.compile(r"^(.*?)/?((?:injected_)?images/.*)$")


def replace_prefix(path, new_prefix, old_prefix=None):
    if old_prefix is not None:
        if not path.startswith(old_prefix):
            raise ValueError(f"Path does not start with --old_prefix: {path}")
        relative = path[len(old_prefix):].lstrip("/")
    else:
        match = IMAGE_DIR_PATTERN.match(path)
        if match is None:
            raise ValueError(f"Cannot find an images/ or injected_images/ folder in path: {path}")
        relative = match.group(2)
    return os.path.join(new_prefix, relative)


def process_json(file_path, new_prefix, old_prefix):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        sample["images"] = [replace_prefix(p, new_prefix, old_prefix) for p in sample.get("images", [])]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return [p for sample in data for p in sample["images"]]


def process_parquet(file_path, new_prefix, old_prefix):
    import pandas as pd

    df = pd.read_parquet(file_path)

    def fix_images(images):
        return [{**img, "image": replace_prefix(img["image"], new_prefix, old_prefix)} for img in images]

    df["images"] = df["images"].apply(fix_images)
    df.to_parquet(file_path)
    return [img["image"] for images in df["images"] for img in images]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new_prefix", type=str, required=True,
                        help="Directory that contains the images/ and injected_images/ folders, e.g. /data/pia_screenshots. "
                             "It can be anywhere; it does not have to be inside this repository.")
    parser.add_argument("--old_prefix", type=str, default=None,
                        help="Old root to strip. If omitted, everything before images/ or injected_images/ is replaced.")
    parser.add_argument("--files", type=str, nargs="+", default=DEFAULT_FILES,
                        help="Data files to rewrite in place (.json or .parquet). Defaults to the SFT and RL training data.")
    parser.add_argument("--no_backup", action="store_true", help="Do not save a .bak copy before rewriting.")
    args = parser.parse_args()

    new_prefix = os.path.abspath(os.path.expanduser(args.new_prefix))
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"[skip] {file_path} not found")
            continue
        if not args.no_backup:
            shutil.copy2(file_path, file_path + ".bak")

        if file_path.endswith(".json"):
            paths = process_json(file_path, new_prefix, args.old_prefix)
        elif file_path.endswith(".parquet"):
            paths = process_parquet(file_path, new_prefix, args.old_prefix)
        else:
            print(f"[skip] {file_path}: unsupported file type")
            continue

        missing = [p for p in paths if not os.path.exists(p)]
        print(f"[done] {file_path}")
        if paths:
            print(f"       e.g. {paths[0]}")
            status = "all found" if not missing else f"{len(missing)} NOT FOUND (e.g. {missing[0]}), check --new_prefix"
            print(f"       {len(paths)} images, {status}")


if __name__ == "__main__":
    main()
