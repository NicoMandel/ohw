import os.path
from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from ultralytics.utils.plotting import Annotator, colors

from ohw.dataset import DisplayLabelsDataset
from ohw.utils import save_image, save_label, print_memory_usage

def parse_args():
    parser = ArgumentParser(description="Script for cropping images of a given resolution to include all bounding boxes.")
    parser.add_argument("input", type=str, help="Location of the image input folder.")
    parser.add_argument("labels", type=str, help="Location of labels folder.")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output folder location. If given, will create subfolder with <crops>")
    parser.add_argument("-c", "--crops", default=1280, type=int, help="Size of crops to be used")
    parser.add_argument("--seed", default=0, type=int, help="Seed for numpy to use when cropping. For reproducibility")
    return parser.parse_args()

def _pointInRect(rect : np.ndarray, pt : np.ndarray) -> bool:
    """
        Function to test if a single point is inside a rectangle.
        By testing if any is outside -> fails on first check
    """
    # check if any of them is outside, then return false
    if (pt[0] < rect[0] or pt[0] > rect[2] or pt[1] < rect[1] or pt[1] > rect[3]): return False
    else: return True


def isinsidecrop(xyxy_crop : np.ndarray, xyxy_bbox : np.ndarray) -> int:
    """
        Function to test whether a bounding box is inside a crop.
        a) test if top left corner is inside
        b) test if bottom right corner is inside crop.
        if none are given -> outside. -1
        if only one is given -> is on edge. 0
        if both are given -> contained. +1
    """
    ins_ct = -1
    # top left corner
    if _pointInRect(xyxy_crop, xyxy_bbox[:2]): ins_ct += 1
    if _pointInRect(xyxy_crop, xyxy_bbox[2:]): ins_ct += 1
    return ins_ct

def get_start_location(xyxy_bbox : np.ndarray, crop_size : int, img_shape) -> np.ndarray:
    """
        Function to generate a starting point for a crop from a bounding box. Generates 2 ints between the starting location of the bbox 
        and the end location of the bbox - 1280
    """
    img_h, img_w = img_shape

    # Calculate valid ranges for width (w_s)
    low_w = max(0, xyxy_bbox[2] - crop_size)
    high_w = min(xyxy_bbox[0], img_w - crop_size)
    
    # Calculate valid ranges for height (h_s)
    low_h = max(0, xyxy_bbox[3] - crop_size)
    high_h = min(xyxy_bbox[1], img_h - crop_size)
    
    # Ensure low < high for width
    if low_w == high_w:
        w_s = img_w - crop_size  # Default to bbox start or within image bounds
    else:
        w_s = np.random.randint(low_w, high_w)
    
    # Ensure low < high for height
    if low_h == high_h:
        h_s =  img_h - crop_size  # Default to bbox start or within image bounds
    else:
        h_s = np.random.randint(low_h, high_h)
    return h_s, w_s

def complete_crop(crop_xyxy : np.ndarray, crop_size : int, all_dets : dict, img_shape : tuple) -> tuple:
    """
        Function to get a crop, with no bounding box on the edge.
        TODO - safeguard here. If too many iterations, drop back and continue with the next item on the bbox list
    """
    redo = True
    i = 0 
    while(redo):
        i += 1
        if (i % 10 == 0):
            print_memory_usage(f"regenerating crop iteration {i}")
        h_s, w_s = get_start_location(crop_xyxy, crop_size, img_shape)
        xyxy_crop = w_s, h_s, w_s + crop_size, h_s + crop_size
        
        redo = False
        # check for every bbox in the image if it is inside the 
        contained_bboxes = []
        for k, xyxy_bb in all_dets.items():
            
            # get indicator where the bbox sits. inside, outside, on border?
            indic = isinsidecrop(xyxy_crop, xyxy_bb)
            
            # if outside don't bother. No action.
            # if even just one is on the border - restart process.
            if indic == 0:
                redo=True
                break

            # if bbox completely inside, add id to list
            elif indic == 1:
                contained_bboxes.append(k)
            
    return (w_s, h_s), contained_bboxes

def _plot_crop(img, crop_start, contained_bboxes : list, detections_dict : dict, crop_size : int = 1280):
    """
        function to plot a crop and the bounding boxes and locations
    """
    # whole image
    crop_img = img.copy()
    ann1 = Annotator(
        img, 
        pil=False,
        font_size=2,
        line_width=2
    )
    [ann1.box_label(box, "OHW", colors(0, bgr=False)) for box in detections_dict.values()]
    
    # crop
    c_x1 = crop_start[0]
    c_y1 = crop_start[1]
    c_x2 = c_x1 + crop_size
    c_y2 = c_y1 + crop_size
    crop  = crop_img[c_y1 : c_y2, c_x1 : c_x2]
    assert crop.shape == (1280, 1280, 3), "Crop not 1280 x 1280. Double check"

    # crop in image
    ann1.box_label([c_x1, c_y1, c_x2, c_y2], "crop", colors(1, bgr=False))
    img_w_bboxes = ann1.result()

    # annotating crop
    ann2 = Annotator(
        np.ascontiguousarray(crop),
        pil=False,
        font_size=2,
        line_width=2
    )
    for bb_id in contained_bboxes:
        bb_dim = detections_dict[bb_id]
        bb = [bb_dim[0] - c_x1, bb_dim[1] - c_y1, bb_dim[2] - c_x1, bb_dim[3] - c_y1]
        ann2.box_label(bb, "OHW", colors(0, bgr=False))
    
    crop_w_bboxes = ann2.result()
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(img_w_bboxes)
    ax2.imshow(crop_w_bboxes)
    plt.suptitle("Crop at x: {} y: {}".format(c_x1, c_y1))
    ax2.set_title("{} detections".format(len(contained_bboxes)))
    plt.show()

def save_crops(img : np.ndarray, crops : dict, all_detections : dict, img_id : str,  cropd_i : Path, cropd_l : Path, crop_size : int = 1280):
    """
        Saving image crop segments
    """
    for i, crop in crops.items():
        dim, bboxes = crop
        img_crop = img[dim[1] : dim[1] + crop_size, dim[0] : dim[0] + crop_size]

        # convert crop label dimensions
        bbox_arr = np.array([all_detections[j] for j in bboxes])
        cr_label = convert_label_dim(dim, bbox_arr, crop_size)
        obj_cls = np.zeros(cr_label.shape[0])
        cr_label = np.c_[obj_cls, cr_label]

        # path manipulation and saving
        img_p = cropd_i / (img_id + f"_{i}.jpg")
        save_image(img_crop, str(img_p))
        label_p = cropd_l / (img_id + f"_{i}.txt")
        save_label(cr_label, label_p)

def convert_label_dim(crop_xy : tuple, bbox_xyxy : np.ndarray, crop_size : int = 1280) -> np.ndarray:
    """
        Function to convert the label dimension of the bboxes from the full image to the crop
    """
    crop_xyxy = np.asarray([bbox_xyxy[:,0] - crop_xy[0], bbox_xyxy[:,1] - crop_xy[1], bbox_xyxy[:,2] - crop_xy[0], bbox_xyxy[:,3] - crop_xy[1]]).T
    crop_xywhn = xyxy2xywhn(crop_xyxy.astype(np.float32), w = crop_size, h = crop_size)
    return crop_xywhn

def crop_images(input_dir : str, label_dir : str = None, output_dir : str = None, crop_size : int = 1280):
    site_dir = os.path.abspath(input_dir)
    label_dir = os.path.abspath(label_dir)
    yds = DisplayLabelsDataset(root=site_dir, img_dir=None, ldir=label_dir)

    # if --output is given
    if output_dir:
        cropdir = Path(os.path.abspath(output_dir))
        cropd_i = cropdir / "images"
        cropd_l = cropdir / "labels"
    
    j = 0
    for img, detections, img_id in tqdm(yds, leave=True):
        j +=1
        print_memory_usage(f"Image: {j}")
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

                # _plot_crop(img, crop_start, contained_bboxes, detections_dict)
            
            # write the crops with suffix k and bounding boxes out to the directory
            print("For image: {} with {} detections, got {} crops.".format(
                img_id, detections.shape[0], len(crops)
            ))
            save_crops(img, crops, detections_dict,  img_id, cropd_i, cropd_l)
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
    np.random.seed(args.seed)
    crop_images(args.input, args.labels, args.output, args.crops)