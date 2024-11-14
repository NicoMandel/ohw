import os.path
from pathlib import Path
from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
import cv2
from ohw.utils import write_summary

# https://docs.ultralytics.com/guides/sahi-tiled-inference/#batch-prediction
def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder. Will count files and instances.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for <json_summary.txt> file. Will write here, otherwise run output to screen.")
    parser.add_argument("--fext", type=str, default=".JSON", help="File extension to be looked for. Case sensitive. defaults to .JSON")
    parser.add_argument("--pixels", type=int, default=1, help="""Half the number of pixels allowed between blobs to still count as connected. Defaults to 1.\
                                                                half because the dilation operation will increase both sides of a blob...""")
    return parser.parse_args()

def cluster_json(data, buff : int =10, connected : int = 1):
    """
        param connected : how many 0's are allowed between the blobs to count as connected.
        Alternative version, only for direct neighborhood:
        ``` from scipy.ndimage import label
            structure = np.ones((3,3)) # Defines 8-connectivity
            labeled_array, num_blobs = label(binary_image, structure=structure) # Label connected components

    """
    data_np = np.asarray([(d['X'], d['Y']) for d in data])
    x_m = data_np[:,0].max()
    y_m = data_np[:,1].max()

    bin_arr = np.zeros((x_m + buff, y_m + buff))
    bin_arr[data_np[:,0], data_np[:,1]] = 1
    bin_arr = bin_arr.astype(dtype=np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilimg = cv2.dilate(bin_arr, kernel, iterations=connected)
    num_blobs, _ = cv2.connectedComponents(dilimg)

    return num_blobs -1

def count_json(args):
    site_dir = Path(os.path.abspath(args.input))
    fnames = [f.stem for f in site_dir.glob("*"+args.fext)]
    summary = {}
    for f in tqdm(fnames):
        fi = site_dir / (f + args.fext)
        with open(fi) as json_file:
            data = json.load(json_file)
        summary[f] = cluster_json(data[f], connected=args.pixels)

    if args.output:
        summaryf = os.path.join(args.output, "json_summary.txt")
        write_summary(summary, summaryf, orig_len = "unknown")
    else:
        [print("File  {} : {} detections".format(k, v)) for k, v in summary.items()]
        
if __name__=="__main__":
    args = parse_args()
    count_json(args)