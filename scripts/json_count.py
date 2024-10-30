import os.path
from pathlib import Path
from argparse import ArgumentParser
import json
from tqdm import tqdm
from ohw.utils import write_summary

# https://docs.ultralytics.com/guides/sahi-tiled-inference/#batch-prediction
def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder. Will count files and instances.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for <json_summary.txt> file. Will write here, otherwise run output to screen.")
    parser.add_argument("--fext", type=str, default=".JSON", help="File extension to be looked for. Case sensitive. defaults to .JSON")
    return parser.parse_args()

def count_json(args,):
    site_dir = Path(os.path.abspath(args.input))
    fnames = [f.stem for f in site_dir.glob("*"+args.fext)]
    summary = {}
    for f in tqdm(fnames):
        fi = site_dir / (f + args.fext)
        with open(fi) as json_file:
            data = json.load(json_file)
        summary[f] = len(data[f])

    if args.output:
        summaryf = os.path.join(args.output, "json_summary.txt")
        write_summary(summary, summaryf, orig_len = "unknown")
    else:
        [print("File  {} : {} detections".format(k, v)) for k, v in summary.items()]
        
if __name__=="__main__":
    args = parse_args()
    count_json(args)