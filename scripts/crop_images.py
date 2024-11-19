import os.path
from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics.utils.ops import xywhn2xyxy
from ohw.dataset import DisplayLabelsDataset
from ohw.utils import save_image, save_label
from display_dataset import annotate_image

def parse_args():
    parser = ArgumentParser(description="Script for cropping images of a given resolution to include all bounding boxes.")
    parser.add_argument("input", type=str, help="Location of the image input folder.")
    parser.add_argument("labels", type=str, help="Location of labels folder.")
    parser.add_argument("name", type=str, default=None, help="Name of the dataset to be used as prefix for the crops.")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output folder location. If given, will create subfolder with <crops>")
    parser.add_argument("-c", "--crops", default=1280, type=int, help="Size of crops to be used")
    parser.add_argument("--seed", default=0, type=int, help="Seed for numpy to use when cropping. For reproducibility")
    return parser.parse_args()

def isinsidecrop(xyxy_crop : np.ndarray, xyxy_bbox : np.ndarray) -> int:
    """
        Function to test whether a bounding box is inside a crop.
        a) test if top left corner is inside
        b) test if bottom right corner is inside crop.
        if none are given -> outside. -1
        if only one is given -> is on edge. 0
        if both are given -> contained. +1
    """
    pass

def get_start_location(xyxy_bbox : np.ndarray, crop_size : int) -> np.ndarray:
    """
        Function to generate a starting point for a crop from a bounding box. Generates 2 ints between the starting location of the bbox 
        and the end location of the bbox - 1280
    """
    x_s = np.random.randint(xyxy_bbox[2] - crop_size, xyxy_bbox[0])
    y_s = np.random.randint(xyxy_bbox[3] - crop_size, xyxy_bbox[1])
    return x_s, y_s

def complete_crop(crop_xyxy) -> tuple:
    """
        Function to get a crop, with no bounding box on the edge
    """


def crop_images(input_dir : str, label_dir : str = None, ds_name : str = None, output_dir : str = None, crop_size : int = 1280):
    site_dir = os.path.abspath(input_dir)
    label_dir = os.path.abspath(label_dir)
    yds = DisplayLabelsDataset(root=site_dir, img_dir=None, ldir=label_dir)

    # if --output is given
    if output_dir:
        cropdir = Path(os.path.abspath(output_dir))
        cropd_i = cropdir / "images"
        cropd_l = cropdir / "labels"
        cropd_i.mkdir(exist_ok=True, parents=True)
        cropd_l.mkdir(exist_ok=True, parents=True)
        print("Created: {}\n and children <images> and <labels>".format(cropdir))
        
    for img, detections, img_id in tqdm(yds, leave=True):
        if np.any(detections):
            crops = {}
            dets = deepcopy(detections)
            
            # unallocated detections
            dets_dict = {i : d for i, d in enumerate(dets)}

            # all detections, by id
            detections_dict = {i : d for i, d in enumerate(detections)}
            
            # crop id
            k = 0

            # while there are still unallocated detections
            while(dets_dict):
                crop_start_det_i = list(dets_dict.keys())[0]
                
                # get the location of the bounding box
                crop_start_det = dets_dict[crop_start_det_i]
                c_s_xyxy = xywhn2xyxy(crop_start_det)
                c_s = get_start_location(c_s_xyxy, crop_size)
                xyxy_crop = c_s[0], c_s[1], c_s[0] + crop_size, c_s[1] + crop_size
                
                # check for every bbox in the image if it is inside the 
                contained_bboxes = []
                for k, xywhn in detections_dict.items():
                    xyxy_bb = xywhn2xyxy(xywhn)
                    indic = isinsidecrop(xyxy_crop, xyxy_bb)
                    
                    # if outside don't bother. No action.
                    # if on the border - restart process.
                    if indic == 0:
                        # todo : restart process from generating crop ids
                        pass
                    # if bbox completely inside, add id to list
                    elif indic == 1:
                        contained_bboxes.append(k)
                

                # crop number. Store top left coordinates and all ids from "detections_dict" which are contained
                crops[k] = tuple(c_s, contained_bboxes)

                # remove all detections that are already allocated to this crop
                [dets_dict.pop(b_id) for b_id in contained_bboxes if b_id in dets_dict]
                k += 1
            
            # write the crops with suffix k and bounding boxes out to the directory
            
            img_w_bboxes = annotate_image(img, detections)

        # if no label -> for false Positives
        else: 
            img_w_bboxes = img

        if output_dir:
            out_p = visdir / (img_id + ".jpg")
            save_image(img_w_bboxes, str(out_p))
        else:  
            plt.imshow(img_w_bboxes)
            plt.show()

if __name__=="__main__":
    args = parse_args()
    np.random.seed(args.seed)
    crop_images(args.input, args.labels, args.name, args.output, args.crops)