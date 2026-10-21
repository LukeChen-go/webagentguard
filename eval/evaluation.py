import ctypes
ctypes.CDLL("libcuda.so")
import argparse
import os
# os.environ["LD_LIBRARY_PATH"] = "/home/yulin/my_libs/cuda_fix:" + os.environ.get("LD_LIBRARY_PATH", "")
import sys
sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import jload
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from utils import Logger
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils import evaluation_metrics

def generate(messages, processor, vllm_model, sampling_params):
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
    prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}

    outputs = vllm_model.generate([llm_inputs], sampling_params=sampling_params)
    response = outputs[0].outputs[0].text
    # print(response)

    return response

def run(args):

    logger = Logger(args.log_file)

    vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.70, max_num_seqs=256,
                     limit_mm_per_prompt={"image": 10, "video": 10})
    sampling_params = SamplingParams(max_tokens=4096)
    processor = AutoProcessor.from_pretrained(args.model_path)


    samples = jload(args.sample_path)
    prediction = []
    gd = []
    unsolved_samples = []
    for sample in tqdm(samples):
        messages = sample["llm_input"]
        label = sample["label"]
        try:
            response = generate(messages, processor, vllm_model, sampling_params)
            logger.log(response)
            
        except Exception as e:
            unsolved_samples.append(messages[1]['content'][0]['image'])
            print(sample)
            print(e)
            continue
        
        try:
            
            answer = response.split("<answer>")[1].split("</answer>")[0]
            answer = answer.strip()
            label = label.strip()
            # logger.log(answer)
            # logger.log(label)
            prediction.append(answer.lower() == "positive")
            gd.append(label.lower() == "positive")
        except Exception as e:
            print(e)
            prediction.append(-1)
            gd.append(label.lower() == "positive")

    # acc = accuracy_score(gd, prediction)
    # recall = recall_score(gd, prediction, zero_division=0)
    # precision = precision_score(gd, prediction, zero_division=0)
    # f1 = f1_score(gd, prediction, zero_division=0)

    # logger.log(f"Accuracy: {acc:.4f}")
    # logger.log(f"Recall: {recall:.4f}")
    # logger.log(f"Precision: {precision:.4f}")
    # logger.log(f"F1 Score: {f1:.4f}")
    results = evaluation_metrics(gd, prediction)
    logger.log(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train/sft/LLaMA-Factory/saves/full/Qwen2.5-VL-3B/SFT-3B'))
    parser.add_argument('--sample_path', type=str, default='./VPI_evaluation.json')
    parser.add_argument('--log_file', type=str, default='logs/eval_log.txt')
    args = parser.parse_args()
    run(args)

