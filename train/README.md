# Training: Prompt Injection Detection (Qwen3-VL-4B)

The detector is trained in two stages:

1. **SFT** with [LLaMA-Factory](sft/LLaMA-Factory) on reasoning traces (`<think> ... </think><answer>positive|negative</answer>`).
2. **RL (GRPO)** with [verl](rl/verl), starting from the SFT checkpoint and using a rule-based reward on the final answer.

Each sample contains a user instruction, a screenshot of the website, and its simplified HTML. The model must decide whether the page contains a prompt injection.

```
train/
├── setup_envs.sh             # creates the conda envs for SFT and RL
├── replace_image_paths.py    # rewrites absolute image paths in the training data
├── sft/LLaMA-Factory/
│   ├── data/detection_sft_data.json    # SFT data (1,921 samples)
│   ├── data/dataset_info.json          # registers `detection_sft_data`
│   └── sft_qwen_3_4b.sh                # SFT launch script
└── rl/verl/
    ├── data/train.parquet, test.parquet               # RL data
    ├── verl/utils/reward_score/detection.py           # reward function
    ├── examples/grpo_trainer/run_qwen3_vl_4b.sh       # RL launch script
    └── merge_fgdp_to_hf.py                            # converts RL checkpoints to HF format
```

## Pipeline

The whole workflow has three steps:

1. **Set up the environments** ([Section 1](#1-set-up-the-environments)): create the `pia-sft` and `pia-rl` conda envs.
2. **Change the image paths** ([Section 2](#2-change-the-image-paths)): point the training data at your copy of the images.
3. **Start training** ([Section 3](#3-start-training)): run SFT first, then RL from the SFT checkpoint.

```bash
cd train

# Step 1: set up the environments
bash setup_envs.sh

# Step 2: change the image paths
conda activate pia-rl
python replace_image_paths.py --new_prefix /path/to/screenshots

# Step 3a: SFT
conda activate pia-sft
cd sft/LLaMA-Factory && bash sft_qwen_3_4b.sh && cd ../..

# Step 3b: RL (edit the paths in run_qwen3_vl_4b.sh first, see Section 3.2)
conda activate pia-rl
cd rl/verl && bash examples/grpo_trainer/run_qwen3_vl_4b.sh
python merge_fgdp_to_hf.py --local_dir checkpoints --experiment_name pia_detection/qwen3-4b-grpo \
    --global_step 321 --base_model_name Qwen/Qwen3-VL-4B-Instruct
```

## 1. Set up the environments

SFT and RL need different versions of torch, so they use two separate conda environments:

| Env | Used for | Key packages |
|---|---|---|
| `pia-sft` | SFT | torch 2.4.1 (CUDA 12.1), LLaMA-Factory, transformers 4.57.1, deepspeed 0.15.4, flash-attn 2.8.2 |
| `pia-rl` | RL | torch 2.8.0, vLLM 0.11.0, verl, transformers 4.57.1, ray 2.50.1, flash-attn 2.8.3 |

```bash
cd train
bash setup_envs.sh        # create both envs
bash setup_envs.sh sft    # only the SFT env
bash setup_envs.sh rl     # only the RL env
```

You can rename the envs with `SFT_ENV=... RL_ENV=... bash setup_envs.sh`. If flash-attn has no prebuilt wheel for your system, it is compiled from source. That needs `nvcc` on `PATH` and can take a long time. Set `MAX_JOBS=4` if you run out of memory.

## 2. Change the image paths

The training data stores image paths with a placeholder root, for example:

```
/path/to/screenshots/images/Aviation/Aviation_7.png
```

The screenshots can live anywhere, such as a separate data disk. Put the `images/` and `injected_images/` folders together under one directory, then rewrite the paths to point there. The script keeps everything from `images/` or `injected_images/` onward and replaces the part before it with `--new_prefix`:

```bash
cd train
conda activate pia-rl     # needs pandas + pyarrow to edit the .parquet files
python replace_image_paths.py --new_prefix /path/to/screenshots
# -> /path/to/screenshots/images/Aviation/Aviation_7.png
```

By default the script updates these files in place and saves a `.bak` copy of each:

- `sft/LLaMA-Factory/data/detection_sft_data.json`
- `rl/verl/data/train.parquet`
- `rl/verl/data/test.parquet`

For each file it prints one example path and how many images could not be found, so you can check the prefix straight away. You can run the script again at any time to move the screenshots to a new location. Other options: `--files` (choose files), `--old_prefix` (strip an exact prefix), `--no_backup`.

## 3. Start training

Training runs in two stages: SFT first, then RL starting from an SFT checkpoint.

### 3.1 SFT

**Hardware:** 2 GPUs by default, with DeepSpeed ZeRO-3 plus CPU offload.

```bash
conda activate pia-sft
cd train/sft/LLaMA-Factory
export WANDB_API_KEY=<your_key>   # see the note below
bash sft_qwen_3_4b.sh
```

Before running, check the top of [sft_qwen_3_4b.sh](sft/LLaMA-Factory/sft_qwen_3_4b.sh):

| Variable | Default | Meaning |
|---|---|---|
| `device` | `"0,1"` | GPUs to use (`CUDA_VISIBLE_DEVICES`) |
| `model` | `Qwen/Qwen3-VL-4B-Instruct` | Base model (HF id or local path) |
| `save_path` | `saves/full/Qwen3-VL-4B/SFT-4B` | Output directory |
| `batch_size` | `8` | Per-device batch size |
| `cutoff_len` | `8192` | Max sequence length |

Main hyperparameters: full fine-tuning, `lr=5e-6` with a cosine schedule and 10% warmup, 4 epochs, bf16, `image_max_pixels=802816`, chat template `qwen3_vl`.

**Notes**

- **W&B key:** the script does not contain a key. Run `wandb login` or `export WANDB_API_KEY=<your_key>` before launching. To turn off W&B, change `--report_to wandb` to `--report_to none`.
- **Model cache:** `HF_HOME=./` downloads the base model into the current directory. Change it if you want a shared cache.
- **Different GPU count:** if you change the number of GPUs, adjust `batch_size` or `--gradient_accumulation_steps` to keep the same effective batch size (16 = 8 × 2 GPUs).

**Output.** A checkpoint is saved after each epoch. With 2 GPUs, one epoch is 121 steps, so the checkpoints are:

```
saves/full/Qwen3-VL-4B/SFT-4B/checkpoint-{121,242,363,484}
```

These are standard Hugging Face checkpoints. The RL stage starts from `checkpoint-363` (the end of epoch 3).

### 3.2 RL (GRPO)

**Hardware:** 2 GPUs by default. FSDP is used for training and vLLM for generating rollouts.

#### Paths in the launch script

[run_qwen3_vl_4b.sh](rl/verl/examples/grpo_trainer/run_qwen3_vl_4b.sh) works out the repo root (`$ROOT_DIR`) from its own location, so these paths work on any machine. Only change them if your files are somewhere else:

| Argument | Set to |
|---|---|
| `data.train_files` | `<repo>/train/rl/verl/data/train.parquet` |
| `data.val_files` | `<repo>/train/rl/verl/data/test.parquet` |
| `actor_rollout_ref.model.path` | Your SFT checkpoint, e.g. `<repo>/train/sft/LLaMA-Factory/saves/full/Qwen3-VL-4B/SFT-4B/checkpoint-363` |
| `custom_reward_function.path` | `<repo>/train/rl/verl/verl/utils/reward_score/detection.py` |

Set `WANDB_API_KEY` the same way as for SFT. To log only to the console, set `trainer.logger='["console"]'`.

#### Run

```bash
conda activate pia-rl
cd train/rl/verl
bash examples/grpo_trainer/run_qwen3_vl_4b.sh
```

Main hyperparameters:

| Setting | Value |
|---|---|
| Algorithm | GRPO, 5 rollouts per prompt, KL loss (`low_var_kl`, coef 0.01), no KL penalty in the reward |
| Batch | 32 prompts per step, PPO mini-batch 16, micro-batch 4 per GPU |
| Lengths | prompt 6144, response 2048 tokens |
| Optimizer | `lr=1e-6`, 3 epochs |
| Rollout | vLLM, TP=1, `gpu_memory_utilization=0.7` |

The reward ([detection.py](rl/verl/verl/utils/reward_score/detection.py)) is binary. It is 1 only when the response contains both `<think> ... </think>` and `<answer> ... </answer>`, and the answer matches the ground-truth label (`positive` / `negative`). Otherwise it is 0.

**Different GPU count:** change `trainer.n_gpus_per_node`. If you run out of memory, lower `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`, or lower `gpu_memory_utilization`.

#### Convert the checkpoint to Hugging Face format

verl writes sharded FSDP checkpoints to `checkpoints/pia_detection/qwen3-4b-grpo/global_step_<N>/actor`. With the default settings, training ends at `global_step_321` and saves a checkpoint there. To merge it into a Hugging Face model:

```bash
cd train/rl/verl
python merge_fgdp_to_hf.py \
    --local_dir checkpoints \
    --experiment_name pia_detection/qwen3-4b-grpo \
    --global_step 321 \
    --base_model_name Qwen/Qwen3-VL-4B-Instruct
```

The merged model and processor are written to `checkpoints/pia_detection/qwen3-4b-grpo/global_step_321/huggingface`. You can load it with `transformers` or serve it with vLLM.
