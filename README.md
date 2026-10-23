# WebAgentGuard

Official implementation of the paper **"WebAgentGuard: A Multimodal Reasoning-Driven Guard
Model for Detecting Prompt Injection Attacks in Web Agents"**, accepted at **WACV 2027**.

WebAgentGuard is a vision-language guard model that detects prompt-injection attacks (PIA)
on web pages. The model reads a browser agent's **user instruction**, a
**screenshot** of the page and the page's **simplified HTML**. It reasons inside
`<think>…</think>` and answers `positive` (injection present) or `negative` (clean) inside
`<answer>…</answer>`.

```
.
├── train/   # SFT (LLaMA-Factory) + RL/GRPO (verl) for Qwen3-VL; see train/README.md
└── eval/    # benchmarks and evaluation scripts; see eval/README.md
```

Released checkpoints: [`Yulin123/guard-qwen3-4b`](https://huggingface.co/Yulin123/guard-qwen3-4b)
and [`Yulin123/guard-qwen3-8b`](https://huggingface.co/Yulin123/guard-qwen3-8b).

## Pipeline

1. [Download the images](#1-download-the-images)
2. [Change the image paths](#2-change-the-image-paths) in the training and evaluation data
3. [Train the model](#3-train-the-model)
4. [Evaluate](#4-evaluate)

## 1. Download the images

The screenshots are not stored in this repo. Download them from Google Drive:

**https://drive.google.com/drive/folders/1I5g6UM-TvFiaq30OqAGqhfPNVKgzq_Y-?usp=sharing**

Put all the image folders together under one directory. It can be anywhere, for example a
separate data disk:

```
/path/to/screenshots/
├── images/                   # Ours (clean)            train + eval
├── injected_images/          # Ours (injected)         train + eval
├── topweb_images/            # Popup + TopWeb (clean)  eval
├── injected_topweb_images/   # Popup + TopWeb (injected) eval
├── VPI_images/               # VPI                     eval
└── EIA_images/               # EIA                     eval
```

If the folders come as archives, unzip each one so the layout matches the tree above.

## 2. Change the image paths

The training and evaluation data store image paths with a placeholder root. Each directory
has a `replace_image_paths.py` script that rewrites them to point at your screenshot
directory. Use the same `--new_prefix` for both:

```bash
# Training data (SFT .json + RL .parquet; needs pandas + pyarrow)
cd train
python replace_image_paths.py --new_prefix /path/to/screenshots
cd ..

# Evaluation data (the four benchmark JSONs)
cd eval
python replace_image_paths.py --new_prefix /path/to/screenshots
cd ..
```

The scripts update these files in place and save a `.bak` copy of each:

| Script | Files updated |
|---|---|
| `train/replace_image_paths.py` | `train/sft/LLaMA-Factory/data/detection_sft_data.json`, `train/rl/verl/data/train.parquet`, `train/rl/verl/data/test.parquet` |
| `eval/replace_image_paths.py` | `eval/evaluation_data.json`, `eval/popup_topweb_evaluation.json`, `eval/VPI_evaluation.json`, `eval/EIA_evaluation.json` |

For each file the script prints an example path and how many images it could not find.
If it reports `NOT FOUND`, check `--new_prefix`. You can rerun the scripts any time you move
the screenshots.

## 3. Train the model

Training runs in two stages: SFT, then RL (GRPO) starting from the SFT checkpoint. Each
stage has its own conda env. Full details (hardware, hyperparameters, W&B setup) are in
[train/README.md](train/README.md).

```bash
cd train

# Set up the pia-sft and pia-rl conda envs
bash setup_envs.sh

# Stage 1: SFT -> saves/full/Qwen3-VL-4B/SFT-4B/checkpoint-*
conda activate pia-sft
cd sft/LLaMA-Factory && bash sft_qwen_3_4b.sh && cd ../..

# Stage 2: RL from SFT checkpoint-363
conda activate pia-rl
cd rl/verl && bash examples/grpo_trainer/run_qwen3_vl_4b.sh

# Merge the final FSDP checkpoint into a Hugging Face model
python merge_fgdp_to_hf.py --local_dir checkpoints --experiment_name pia_detection/qwen3-4b-grpo \
    --global_step 321 --base_model_name Qwen/Qwen3-VL-4B-Instruct
# -> checkpoints/pia_detection/qwen3-4b-grpo/global_step_321/huggingface
```

## 4. Evaluate

Set up the evaluation env and run `evaluation_batch.py` on each benchmark. Full details
(benchmarks, metrics, log format) are in [eval/README.md](eval/README.md).

```bash
conda create -n pia-eval python=3.10 -y
conda activate pia-eval
pip install torch==2.8.0 vllm==0.11.0 transformers==4.57.1 qwen-vl-utils==0.0.14 \
            accelerate scikit-learn openai tqdm

cd eval
mkdir -p logs
MODEL=../train/rl/verl/checkpoints/pia_detection/qwen3-4b-grpo/global_step_321/huggingface
# or a released checkpoint: MODEL=Yulin123/guard-qwen3-4b

for bench in evaluation_data popup_topweb_evaluation VPI_evaluation EIA_evaluation; do
  python evaluation_batch.py \
    --model_path $MODEL \
    --sample_path ./${bench}.json \
    --log_file logs/${bench}.txt
done

# The last line of each log is the metrics dict
for f in logs/*.txt; do echo "$f"; tail -n 1 "$f"; done
```

| Benchmark | File | #Samples | Pos / Neg |
|---|---|---:|---|
| Ours (synthetic websites) | `evaluation_data.json` | 1000 | 500 / 500 |
| Popup + TopWeb | `popup_topweb_evaluation.json` | 2021 | 969 / 1052 |
| VPI | `VPI_evaluation.json` | 306 | 306 / 0 |
| EIA | `EIA_evaluation.json` | 1415 | 1415 / 0 |

VPI and EIA contain only attacks, so report recall/accuracy for them rather than precision.
