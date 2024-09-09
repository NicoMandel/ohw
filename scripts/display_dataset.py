import os.path
from pathlib import Path
from argparse import ArgumentParser
# from ultralytics.data import YOLODataset
from ultralytics.utils.plotting import Annotator, colors
from ohw.dataset import DisplayLabelsDataset
from ohw.utils import det_to_bb, save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure with <visualisations> subfolder in subdirectory.")
    args = parser.parse_args()
    return vars(args)

if __name__=="__main__":
    args = parse_args()
    fdir = os.path.abspath(os.path.dirname(__file__))
    idir = "images"
    ldir = "labels"

    site_dir = os.path.abspath(os.path.expanduser("~/src/csu/data/OHW/0.24_sites/CH-NE"))
    visdir = Path(os.path.abspath(os.path.join(site_dir, "visualisations")))
    yds = DisplayLabelsDataset(site_dir)

    # if --output is given
    if args.output:
        visdir.mkdir(exist_ok=True)
        print("Created: {}\nWill write images to it, if they don't exist yet".format(visdir))

    
    for img, detections, img_id in tqdm(yds):
        ann = Annotator(
            img,
            line_width=None,  # default auto-size
            font_size=None,  # default auto-size
            font="Arial.ttf",  # must be ImageFont compatible
            pil=False,  # use PIL, otherwise uses OpenCV
        )
        bboxes = det_to_bb(img.shape, detections)
        for nb, box in enumerate(bboxes):
            c_idx, *box = box
            label = "OHW"
            ann.box_label(box, label, colors(c_idx, bgr=True))
        img_w_bboxes = ann.result()
        if output:
            out_p = visdir / (img_id + ".jpg")
            save_image(img_w_bboxes, str(out_p))
        else:  
            plt.imshow(img_w_bboxes)
            plt.show()
