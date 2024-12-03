# https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.autosplit
# auto-labelling-> https://docs.ultralytics.com/usage/simple-utilities/?h=sam+labeling

import os.path
from argparse import ArgumentParser
from ultralytics.data.utils import autosplit

def parse_args():
    parser = ArgumentParser(description="Script for generating new splits, after new data has been added to dataset.")
    parser.add_argument("input", type=str, help="Location of the dataset image folder to be split. **Excluding** the <train> part in the end, for correct file placement.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    autosplit(
        path=args.input,
        weights=(0.8, 0.2, 0.0),
        annotated_only=False
    )