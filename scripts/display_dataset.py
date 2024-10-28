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
    parser.add_argument("input", type=str, help="Location of the input folder, as root. If optional argument <labels> is not given, will find <images> and <labels> subfolders. Otherwise uses this as the input folder.")
    parser.add_argument("-l", "--labels", default=None, type=str, help="Location of labels folder, if not subfolder of root. If not given, will look for <images> and <labels> subfolder in argument <input>.")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output folder location. If given, will create subfolder with <visualisations>")
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the dataset to be used inside the <output>/dataset/<visualisations>. If none given, will use folder name of ds.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Boolean value. If given, will look recursively for subfolders <images> and <labels> and add them to the set.")
    return parser.parse_args()

def annotate_image(img : np.ndarray, detections : np.ndarray, line_width : int = None, font_size : int =None) -> np.ndarray:
        # https://docs.ultralytics.com/usage/simple-utilities/#horizontal-bounding-boxes
        ann = Annotator(
            img,
            line_width=line_width,  # default auto-size
            font_size=font_size,  # default auto-size
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

def visualise_images(input_dir : str, label_dir : str = None, output_dir : str = None, output_name : str = None):
    site_dir = os.path.abspath(input_dir)
    outdir_name =  output_name if output_name else os.path.basename(os.path.normpath(site_dir))
    yds = DisplayLabelsDataset(site_dir) if label_dir is None else DisplayLabelsDataset(root=site_dir, img_dir=None, ldir=label_dir)

    # if --output is given
    if output_dir:
        visdir = Path(os.path.abspath(os.path.join(output_dir, outdir_name, "visualisations")))
        visdir.mkdir(exist_ok=True, parents=True)
        print("Created: {}\nWill write images to it, if they don't exist yet".format(visdir))
        
    for img, detections, img_id in tqdm(yds, leave=True):
        if np.any(detections):
            img_w_bboxes = annotate_image(img, detections)

        # if no label -> for false Positives
        else: 
            img_w_bboxes = img
            # continue

        if output_dir:
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
            site_n = os.path.basename(os.path.normpath(str(site)))
            visualise_images(str(site), args.output, output_name=site_n)
    else:
        visualise_images(args.input, args.labels, args.output, args.name)