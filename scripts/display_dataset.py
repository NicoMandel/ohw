import os.path
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from ohw.dataset import DisplayLabelsDataset
from ohw.utils import det_to_bb, save_image, get_site_dirs
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure with <visualisations> subfolder in subdirectory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Boolean value. If given, will look recursively for subfolders <images> and <labels> and add them to the set.")
    return parser.parse_args()

def annotate_image(img : np.ndarray, detections : np.ndarray) -> np.ndarray:
        # https://docs.ultralytics.com/usage/simple-utilities/#horizontal-bounding-boxes
        ann = Annotator(
            img,
            line_width=None,  # default auto-size
            font_size=None,  # default auto-size
            font="Arial.ttf",  # must be ImageFont compatible
            pil=False,  # use PIL, otherwise uses OpenCV
        )
        bboxes = det_to_bb(img.shape, detections)
        for box in bboxes:
            c_idx, *box = box
            label = "OHW"
            ann.box_label(box, label, colors(c_idx, bgr=False))
        img_w_bboxes = ann.result()
        return img_w_bboxes

def visualise_images(input_dir : str, output : bool):
    site_dir = os.path.abspath(input_dir)
    visdir = Path(os.path.abspath(os.path.join(site_dir, "visualisations")))
    yds = DisplayLabelsDataset(site_dir)

    # if --output is given
    if output:
        visdir.mkdir(exist_ok=True)
        print("Created: {}\nWill write images to it, if they don't exist yet".format(visdir))
        
    for img, detections, img_id in tqdm(yds, leave=True):
        img_w_bboxes = annotate_image(img, detections)
        if output:
            out_p = visdir / (img_id + ".jpg")
            save_image(img_w_bboxes, str(out_p))
        else:  
            plt.imshow(img_w_bboxes)
            plt.show()

if __name__=="__main__":
    args = parse_args()
    if args.recursive:
        site_dirs = get_site_dirs(args.input)
        for site in site_dirs:
            visualise_images(str(site), args.output)
    else:
        visualise_images(args.input, args.output)