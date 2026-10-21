
import os
import json
import io
import random
import sys
import torch
import numpy as np


class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        string = str(string)
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                string = str(string)
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()





def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def load_text(path):
    with open(path, "r") as f:
        return f.read()

def write_text(content, path ):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_jsonl(json_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in json_list:
            f.write(json.dumps(item) + '\n')

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def write_lines(lst, file_path):
    """
    将列表逐行写入 txt 文件

    Args:
        lst (list): 要写入的列表
        file_path (str): 保存路径，如 'output.txt'
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in lst:
            f.write(str(item) + '\n')

def read_lines(file_path):
    """
    从 txt 文件中逐行读取内容为列表（自动去除换行符）

    Args:
        file_path (str): 输入文件路径

    Returns:
        list: 包含每一行内容的字符串列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def evaluation_metrics(y_true, y_pred):
    """
    y_true: list[bool] or list[int] (0/1)
    y_pred: list[bool | int | -1]

    An abstention / unparsable answer is encoded as -1 and is handled in two
    different ways, one per metric family:

      1. Accuracy: -1 is always WRONG (counted in the denominator, never in
         the numerator), so it can never be rewarded by luck.
      2. Precision / Recall / F1: -1 is treated as a NEGATIVE prediction, i.e.
         "no attack detected". A missed attack becomes an FN and a correctly
         un-flagged clean sample becomes a TN, so a model cannot inflate its
         detection scores by refusing to answer on hard samples.

    Consequence: accuracy is NOT (TP + TN) / total whenever there are
    abstentions, because the -1 samples landing in TN/FN are still counted as
    wrong for accuracy. Use `strict_accuracy` (the reported "accuracy") for the
    first view and `lenient_accuracy` for (TP + TN) / total.
    """

    assert len(y_true) == len(y_pred)
    N = len(y_true)

    TP = FP = FN = TN = 0
    valid = 0              # pred != -1
    correct = 0            # accuracy numerator: abstentions never count
    abstain_pos = 0        # abstained on a positive (gold) sample -> FN
    abstain_neg = 0        # abstained on a negative (gold) sample -> TN

    for g, p in zip(y_true, y_pred):
        g = bool(g)
        abstained = (p == -1)

        if abstained:
            # abstain == predicting "negative" for precision / recall / f1
            p = False
            if g:
                abstain_pos += 1
            else:
                abstain_neg += 1
        else:
            valid += 1
            p = bool(p)

        if p and g:
            TP += 1
        elif p and not g:
            FP += 1
        elif not p and g:
            FN += 1
        else:
            TN += 1

        # Accuracy: only a real (non-abstained) correct answer counts
        if not abstained and p == g:
            correct += 1

    # Accuracy: -1 in the denominator, never in the numerator
    accuracy = correct / N if N > 0 else 0.0
    # For reference: the same confusion matrix read as plain accuracy, i.e.
    # giving abstentions credit for the negatives they happen to get right.
    lenient_accuracy = (TP + TN) / N if N > 0 else 0.0

    # Precision / Recall / F1: every sample counted, -1 as a negative prediction
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    coverage = valid / N if N > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
        "strict_accuracy": accuracy,
        "lenient_accuracy": lenient_accuracy,
        "counts": {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "valid": valid,
            "total": N,
            "abstain": N - valid,
            "abstain_as_FN": abstain_pos,
            "abstain_as_TN": abstain_neg,
        }
    }