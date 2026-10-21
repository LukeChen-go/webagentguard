# Evaluating Prompt-Injection Guards on Web Benchmarks

This directory holds the evaluation benchmarks and scripts for the prompt-injection
attack (PIA) detectors. Each detector looks at a browser agent's **user instruction**, a
**screenshot** of the page and the page's **processed HTML**. It reasons inside
`<think>…</think>` and outputs `positive` (injection present) or `negative` (clean)
inside `<answer>…</answer>`.

Released checkpoints (both are full `Qwen3VLForConditionalGeneration` weights, so no
adapter merging is needed):

| Checkpoint | Base model | Hugging Face |
|---|---|---|
| Guard-Qwen3-4B | Qwen3-VL-4B | [`Yulin123/guard-qwen3-4b`](https://huggingface.co/Yulin123/guard-qwen3-4b) |
| Guard-Qwen3-8B | Qwen3-VL-8B | [`Yulin123/guard-qwen3-8b`](https://huggingface.co/Yulin123/guard-qwen3-8b) |

---

## 1. Installation

Tested with:

| Package | Version |
|---|---|
| python | 3.10 |
| CUDA | 12.4 |
| torch | 2.8.0 |
| vllm | 0.11.0 |
| transformers | 4.57.1 |
| qwen-vl-utils | 0.0.14 |

```bash
conda create -n pia-eval python=3.10 -y
conda activate pia-eval
pip install torch==2.8.0 vllm==0.11.0 transformers==4.57.1 qwen-vl-utils==0.0.14 \
            accelerate scikit-learn openai tqdm
```

Notes:
- **Point the image paths at your copy of the screenshots first**
  ([Section 2](#2-change-the-image-paths)). After that the scripts can run from any directory.
- `evaluation.py` and `evaluation_batch.py` load `libcuda.so` at import time. They need a machine
  with an NVIDIA GPU and driver.
- Checkpoints download into `HF_HOME` (default `~/.cache/huggingface`). The 8B model is
  about 17 GB. To use a bigger disk:
  ```bash
  export HF_HOME=/path/to/large/disk/hf_cache
  ```
- The Guard-Qwen3 checkpoints are public, so no Hugging Face login is needed.

---

## 2. Change the image paths

The screenshots do not have to be in this directory; they can live anywhere, such as a separate
data disk. Put the six image folders together under one directory:

```
/path/to/screenshots/
├── images/                   # Ours (clean)
├── injected_images/          # Ours (injected)
├── topweb_images/            # Popup + TopWeb (clean)
├── injected_topweb_images/   # Popup + TopWeb (injected)
├── VPI_images/               # VPI
└── EIA_images/               # EIA
```

The benchmark JSONs store screenshot paths relative to this directory (for example
`./EIA_images/EIA/1053.png`). Rewrite them to point at your screenshot directory:

```bash
python replace_image_paths.py --new_prefix /path/to/screenshots
# -> /path/to/screenshots/EIA_images/EIA/1053.png
```

If the screenshots are already inside this directory, use `--new_prefix .`, or skip this step
and run the evaluation scripts from here.

Everything before the image folder (`images/`, `injected_images/`, `injected_topweb_images/`,
`topweb_images/`, `VPI_images/` or `EIA_images/`) is replaced with `--new_prefix`. By default
the script updates these files in place and saves a `.bak` copy of each:

- `evaluation_data.json`
- `popup_topweb_evaluation.json`
- `VPI_evaluation.json`
- `EIA_evaluation.json`

For each file it prints one example path and how many images could not be found, so you can
check the prefix straight away. You can run the script again at any time to move the
screenshots to a new location. Other options: `--files` (choose files), `--old_prefix`
(strip an exact prefix), `--no_backup`.

---

## 3. Benchmarks

Every benchmark is a JSON list of
`{"llm_input": [...chat messages...], "label": "positive" | "negative"}`.
`llm_input` already holds the system prompt, the screenshot path and the user instruction
plus HTML, so the same file works for every model.

| Benchmark | File | #Samples | Pos / Neg | Screenshots | Description |
|---|---|---:|---|---|---|
| **Ours** (synthetic websites) | `evaluation_data.json` | 1000 | 500 / 500 | `injected_images/`, `images/` | Held-out split of our generated websites, clean and injected |
| **Popup + TopWeb** | `popup_topweb_evaluation.json` | 2021 | 969 / 1052 | `injected_topweb_images/`, `topweb_images/` | Pop-up injections rendered onto top-ranked real websites, mixed with the clean originals |
| **VPI** | `VPI_evaluation.json` | 306 | 306 / 0 | `VPI_images/` | Visual Prompt Injection benchmark (attacks only) |
| **EIA** | `EIA_evaluation.json` | 1415 | 1415 / 0 | `EIA_images/` | Environmental Injection Attack benchmark (attacks only) |

> VPI and EIA contain only attacks, so precision is always 1.0 on them. Report
> **recall/accuracy** for these two (on an all-positive set they are the same number
> when the model gives no abstentions).

---

## 4. Quick start: evaluate the released checkpoints

`evaluation_batch.py` is the recommended script. It sends a batch of
samples (default 256) to vLLM at once, which is much faster than `evaluation.py`.
`evaluation.py` runs one sample at a time. Both use the same prompt, parsing and metrics.

```bash
# Guard-Qwen3-4B on EIA
python evaluation_batch.py \
    --model_path Yulin123/guard-qwen3-4b \
    --sample_path ./EIA_evaluation.json \
    --log_file logs/guard_4b_eia.txt

# Guard-Qwen3-8B on VPI
python evaluation_batch.py \
    --model_path Yulin123/guard-qwen3-8b \
    --sample_path ./VPI_evaluation.json \
    --log_file logs/guard_8b_vpi.txt
```

### Run both checkpoints on every benchmark

```bash
#!/bin/bash
# run_guard_all.sh — run from this directory
set -e
declare -A BENCH=(
  [ours]=evaluation_data.json
  [popup_topweb]=popup_topweb_evaluation.json
  [vpi]=VPI_evaluation.json
  [eia]=EIA_evaluation.json
)
mkdir -p logs
for size in 4b 8b; do
  for name in "${!BENCH[@]}"; do
    python evaluation_batch.py \
      --model_path Yulin123/guard-qwen3-${size} \
      --sample_path ./${BENCH[$name]} \
      --log_file logs/guard_${size}_${name}.txt
  done
done
```

To evaluate your own checkpoint, pass any local directory in Hugging Face format as
`--model_path`:

```bash
python evaluation_batch.py --model_path /path/to/your/checkpoint \
    --sample_path ./VPI_evaluation.json --log_file logs/my_model_vpi.txt
```

Arguments:

| Arg | Meaning |
|---|---|
| `--model_path` | HF repo ID or local directory |
| `--sample_path` | One of the benchmark JSONs in Section 3 |
| `--log_file` | Where raw generations and the final metrics are written |
| `--batch_size` | (`evaluation_batch.py` only) number of samples per vLLM call, default 256 |

vLLM is set up with `gpu_memory_utilization=0.70` and `max_tokens=4096`, and the sampling
parameters are left at their defaults. Each checkpoint runs on a single GPU. If you have
less GPU memory, lower `--batch_size` or raise `gpu_memory_utilization` in the script.

---

## 5. Output and metrics

Each run writes every raw model response to `--log_file`. The **last line** is a metrics
dict produced by `utils.evaluation_metrics`:

```python
{'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ..., 'coverage': ...,
 'strict_accuracy': ..., 'lenient_accuracy': ...,
 'counts': {'TP': ..., 'FP': ..., 'FN': ..., 'TN': ..., 'valid': ..., 'total': ...,
            'abstain': ..., 'abstain_as_FN': ..., 'abstain_as_TN': ...}}
```

Labels: `positive` = injection (the positive class), `negative` = clean.

An **abstention** is a response with no parseable `<answer>` tag. It is handled as follows:
- **Accuracy (`strict_accuracy`)**: an abstention always counts as wrong.
- **Precision / Recall / F1**: an abstention is treated as a *negative* prediction, so a
  missed attack counts as an FN.
- `lenient_accuracy` = (TP + TN) / N, which gives abstentions credit for the negatives they
  happen to get right.
- `coverage` = the fraction of samples with a parseable answer.

To print only the metrics from a set of logs:

```bash
for f in logs/*.txt; do echo "$f"; tail -n 1 "$f"; done
```

**Log files are never overwritten.** If `--log_file` already exists, `Logger` appends `+` to
the name (`guard_4b_vpi.txt+`, `guard_4b_vpi.txt++`, …). The file with the most `+`
characters is the newest run.

---

## 6. Directory layout

```
.
├── evaluation.py / evaluation_batch.py   # vLLM evaluation of Guard-Qwen3 checkpoints
├── utils.py                              # jload, Logger, evaluation_metrics
├── replace_image_paths.py                # rewrites the image paths in the benchmark JSONs
├── evaluation_data.json                  # Ours
├── popup_topweb_evaluation.json          # Popup + TopWeb
├── VPI_evaluation.json                   # VPI
└── EIA_evaluation.json                   # EIA (attack only)

/path/to/screenshots/                     # can be anywhere; see Section 2
└── images/, injected_images/, injected_topweb_images/, topweb_images/, VPI_images/, EIA_images/
```
