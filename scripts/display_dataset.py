import os.path
from argparse import ArgumentParser
# from ultralytics.data import YOLODataset
from ultralytics.utils.plotting import Annotator, colors
from ohw.dataset import DisplayLabelsDataset
from ohw.utils import det_to_bb
import cv2
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser(description="Script for displaying the images.")
    parser.add_argument("-i", "--input", required=False, type=str, help="Location of the input folder")
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure")
    args = parser.parse_args()
    return vars(args)

def show_images(input_dir : str):
    pass


if __name__=="__main__":
    # args = parse_args()
    fdir = os.path.abspath(os.path.dirname(__file__))
    idir = "images"
    ldir = "labels"
    site_dir = os.path.abspath(os.path.expanduser("~/src/csu/data/OHW/0.24_sites/CH-NE"))

    yds = DisplayLabelsDataset(site_dir)
    
    for img, detections in yds:
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
            ann.box_label(box, label)
        img_w_bboxes = ann.result()
        plt.imshow(img_w_bboxes)
        plt.show()
        print(type(img))