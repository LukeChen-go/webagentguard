import re



def format_reward(solution_str: str) -> float:
    think_pattern = "<think>" in solution_str and "</think>" in solution_str
    answer_patter = "<answer>" in solution_str and "</answer>" in solution_str
    return 1.0 if think_pattern and answer_patter else 0.0


def acc_reward(solution_str: str, ground_truth: str) -> float:
    try:
        answer = solution_str.split("<answer>")[1].split("</answer>")[0].strip()
        ground_truth = ground_truth.strip()
        return 1.0 if answer == ground_truth else 0.0
    except Exception:
        return 0.0

    


def compute_score(solution_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1, **kwargs) -> float:
    return format_reward(solution_str)  * acc_reward(solution_str, ground_truth) 
