import os.path
from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics.utils.ops import xywhn2xyxy

from ohw.dataset import GPSDataset

def parse_args():
    parser = ArgumentParser(description="Script for cropping images of a given resolution to include all bounding boxes.")
    parser.add_argument("input", type=str, help="Location of the image input folder.")
    parser.add_argument("geotag_csv", type=str, help="Location of the csv to be used for geotagging.")

    parser.add_argument("--kml", action="store_true", help="Name of the dataset to be used as prefix for the crops.")
    return parser.parse_args()

def geotag_images(input_dir : str, geotag_csv : str = None, kml : bool = False):
    site_dir = os.path.abspath(input_dir)
    gps_ds = GPSDataset(root=site_dir, csv_file=geotag_csv)
        
    for img, detections, img_id in tqdm(gps_ds, leave=True):
        if np.any(detections):

            img_shape = img.shape[:2]
            # convert detections to xyxy
            img_h, img_w = img_shape
            detections = xywhn2xyxy(detections[...,1:], img_w, img_h).astype(np.uint16)
            # sort detections
            if len(detections.shape) == 1: detections = detections[np.newaxis,:]
            indic = np.lexsort((detections[:,1], detections[:,0]))
            det_sort = detections[indic]
            
            crops = {}
            dets = deepcopy(det_sort)
            
            # unallocated detections
            dets_dict = {i : d for i, d in enumerate(dets)}

            # all detections, by id
            detections_dict = {i : d for i, d in enumerate(det_sort)}
            
            # crop id
            k = 0

            # while there are still unallocated detections
            while(dets_dict):
                crop_start_det_i = list(dets_dict.keys())[0]
                
                # get the location of the bounding box
                crop_start_det = dets_dict[crop_start_det_i]
                
                # get a crop that ensures that there is no bbox 
                crop_start, contained_bboxes = complete_crop(crop_start_det, crop_size, detections_dict, img_shape)

                # crop number. Store top left coordinates and all ids from "detections_dict" which are contained
                crops[k] = (crop_start, contained_bboxes)

                # remove all detections that are already allocated to this crop
                [dets_dict.pop(b_id) for b_id in contained_bboxes if b_id in dets_dict]
                k += 1

                _plot_crop(img, crop_start, contained_bboxes, detections_dict)
            
            # write the crops with suffix k and bounding boxes out to the directory
            print("For image: {} with {} detections, got {} crops.".format(
                img_id, detections.shape[0], len(crops)
            ))
            save_crops(img, crops, cropd_i, cropd_l)
            # img_w_bboxes = annotate_image(img, detections)

        # # if no label -> for false Positives
        # else: 
        #     img_w_bboxes = img

        # if output_dir:
        #     out_p = visdir / (img_id + ".jpg")
        #     save_image(img_w_bboxes, str(out_p))
        # else:  
        #     plt.imshow(img_w_bboxes)
        #     plt.show()

if __name__=="__main__":
    args = parse_args()
    geotag_images(args.input, args.geotag_csv, args.kml)