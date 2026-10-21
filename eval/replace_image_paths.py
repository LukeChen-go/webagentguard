"""Rewrite the image paths in the evaluation data to point at your copy of the screenshots.

The evaluation data stores image paths such as
    ./EIA_images/EIA/1053.png
Everything from the image folder (images/, injected_images/, injected_topweb_images/,
topweb_images/, VPI_images/ or EIA_images/) onwards is kept, and the part before it is
replaced with the prefix you provide:
    <new_prefix>/EIA_images/EIA/1053.png

<new_prefix> is the directory that contains the image folders. It can be anywhere,
e.g. a separate data disk; it does not have to be this directory.

Supported files:
    *.json  evaluation data  (field: "llm_input": [{"content": [{"type": "image", "image": path}, ...]}, ...])

Usage:
    python replace_image_paths.py --new_prefix /path/to/screenshots
    python replace_image_paths.py --new_prefix /path/to/screenshots --files VPI_evaluation.json
"""

import argparse
import json
import os
import re
import shutil

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = [
    os.path.join(EVAL_DIR, "evaluation_data.json"),
    os.path.join(EVAL_DIR, "popup_topweb_evaluation.json"),
    os.path.join(EVAL_DIR, "VPI_evaluation.json"),
    os.path.join(EVAL_DIR, "EIA_evaluation.json"),
]
IMAGE_DIRS = ["images", "injected_images", "injected_topweb_images", "topweb_images", "VPI_images", "EIA_images"]
# Splits "<old_prefix>/<image_dir>/<rest>" at the first image folder.
IMAGE_DIR_PATTERN = re.compile(r"^(?:(.*?)/)?((?:" + "|".join(IMAGE_DIRS) + r")/.*)$")


def replace_prefix(path, new_prefix, old_prefix=None):
    if old_prefix is not None:
        if not path.startswith(old_prefix):
            raise ValueError(f"Path does not start with --old_prefix: {path}")
        relative = path[len(old_prefix):].lstrip("/")
    else:
        match = IMAGE_DIR_PATTERN.match(path)
        if match is None:
            raise ValueError(f"Cannot find any of {', '.join(d + '/' for d in IMAGE_DIRS)} in path: {path}")
        relative = match.group(2)
    return os.path.join(new_prefix, relative)


def process_json(file_path, new_prefix, old_prefix):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = []
    for sample in data:
        for message in sample["llm_input"]:
            if not isinstance(message["content"], list):
                continue
            for item in message["content"]:
                if item.get("type") == "image":
                    item["image"] = replace_prefix(item["image"], new_prefix, old_prefix)
                    paths.append(item["image"])
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new_prefix", type=str, required=True,
                        help="Directory that contains the image folders (images/, injected_images/, injected_topweb_images/, "
                             "topweb_images/, VPI_images/, EIA_images/), e.g. /data/pia_screenshots. "
                             "It can be anywhere; it does not have to be this directory.")
    parser.add_argument("--old_prefix", type=str, default=None,
                        help="Old root to strip. If omitted, everything before the image folder is replaced.")
    parser.add_argument("--files", type=str, nargs="+", default=DEFAULT_FILES,
                        help="Evaluation JSON files to rewrite in place. Defaults to the four benchmarks.")
    parser.add_argument("--no_backup", action="store_true", help="Do not save a .bak copy before rewriting.")
    args = parser.parse_args()

    new_prefix = os.path.abspath(os.path.expanduser(args.new_prefix))
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"[skip] {file_path} not found")
            continue
        if not file_path.endswith(".json"):
            print(f"[skip] {file_path}: unsupported file type")
            continue
        if not args.no_backup:
            shutil.copy2(file_path, file_path + ".bak")

        paths = process_json(file_path, new_prefix, args.old_prefix)
        missing = [p for p in paths if not os.path.exists(p)]

        print(f"[done] {file_path}")
        if paths:
            print(f"       e.g. {paths[0]}")
            status = "all found" if not missing else f"{len(missing)} NOT FOUND (e.g. {missing[0]}), check --new_prefix"
            print(f"       {len(paths)} images, {status}")


if __name__ == "__main__":
    main()
