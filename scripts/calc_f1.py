import os.path
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Script for calculating the F1 score for the best model")
    parser.add_argument("input", type=str, help="Location of the results csv file.")
    parser.add_argument("--epoch", type=str, default=None, help="Which epoch to calculate F1 score from.")
    # parser.add_argument("--fext", type=str, default=".JSON", help="File extension to be looked for. Case sensitive. defaults to .JSON")
    return parser.parse_args()

def calc_f1(args, eps :float =1e-16):
    """
        https://github.com/ultralytics/yolov5/issues/8701 - maximum fitness is mAP50:0.95
        https://github.com/ultralytics/ultralytics/issues/14137#issuecomment-2201432148
        F1 score: https://github.com/ultralytics/ultralytics/blob/f2a7a29e531ad029255c8ec180ff65de24f42e8d/ultralytics/yolo/utils/metrics.py#L397

        somehow get the confidence value where the maximum is from here:
        https://github.com/ultralytics/ultralytics/blob/f2a7a29e531ad029255c8ec180ff65de24f42e8d/ultralytics/yolo/utils/metrics.py#L346

    """
    df = pd.read_csv(args.input, index_col=0)
    df.columns = df.columns.str.strip()
    df['fitness'] = df["metrics/mAP50(B)"] * 0.1 + df["metrics/mAP50-95(B)"] * 0.9
    best_epoch = df['fitness'].idxmax() + 1

    # all values at best epoch
    ser = df.iloc[best_epoch,:]
    p = ser["metrics/precision(B)"]
    r = ser["metrics/recall(B)"]
    f1 = 2 * p * r  / (p + r + eps)

    return f1
        
if __name__=="__main__":
    args = parse_args()
    calc_f1(args)