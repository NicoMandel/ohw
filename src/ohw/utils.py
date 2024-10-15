from typing import Tuple
import os.path
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
# import rawpy
# from PIL import Image
import cv2
from ultralytics.utils.ops import xywhn2xyxy

def load_image(imgf: str, convert : bool = False):
    bgr_im = cv2.imread(imgf)
    im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB) if convert else bgr_im
    # rgb_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def save_image(img : np.ndarray, path, cvtcolor : bool = False):
    """
        Function to save the iamge.
    """
    img = img if not cvtcolor else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)  

def load_label(label_f : str):
    return np.genfromtxt(label_f, delimiter=' ')

def save_label(array : np.ndarray, label_f : str) -> None:
    """
        TODO: Change, see:
        https://numpy.org/doc/2.0/reference/generated/numpy.savetxt.html
        https://github.com/ultralytics/ultralytics/issues/2143

    """
    with open(label_f, '+w') as f:
        for l in array:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(l[0].astype(int), l[1], l[2], l[3], l[4]))

def get_site_dirs(path : str) -> bool:
    """
        Function to return site dirs by looking for <images> and <labels> subdir.
    """
    if not os.path.isdir(path): return False
    root = Path(path)
    img_set = set([pt.parent.absolute() for pt in root.rglob('**/images/')])
    lab_set = set([pt.parent.absolute() for pt in root.rglob('**/labels/')])
    inters = img_set.intersection(lab_set)
    return inters

def _has_img_subdir(path : str) -> bool:
    pass

def _has_label_subdir(path : str) -> bool:
    pass

def det_to_bb(shape : tuple, detections : np.ndarray) -> list:
    """
        centroid x, centroid y, bb width, bb height
        correct covnersion: https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291 
        https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format 
        or this for simplicity: https://docs.ultralytics.com/usage/simple-utilities/#get-bounding-box-dimensions
        or even better, this one: https://docs.ultralytics.com/usage/simple-utilities/#bounding-boxes
        look through all of this for simplicity!
    """
    if len(detections.shape) == 1:
        detections = detections[np.newaxis,:]
    img_w, img_h = shape[:2]
    det_xyxy = xywhn2xyxy(detections[:,1:], img_w, img_h)
    det = np.c_[detections[:,0], det_xyxy]
    return det

def append_to_xlsx(key, results_dict : dict, xlsx_f : str) -> bool:
    df = pd.DataFrame(
        data=results_dict,
        index=[key]
    )
    try:
        if os.path.exists(xlsx_f):
            df_prev = pd.read_excel(open(xlsx_f, 'rb'), header=0)
            result = pd.concat([df_prev, df])
        else:
            result = df
        with pd.ExcelWriter(xlsx_f) as writer:
            result.to_excel(writer, index=0)
        return True
    except OSError as ose:
        print(ose)
        return False
    
# def load_arw(fpath : str) -> np.ndarray:
#     """
#         Loading an ARW image. Thanks to Rob
#     """
#     raw = rawpy.imread(fpath)
#     return raw.postprocess(use_camera_wb=True, output_bps=8)

# def read_arw_as_pil(fpath : str) -> Image.Image:
#     """ 
#         function to return an ARW image in PIL format for SAHI
#     """
#     np_arr = load_arw(fpath)
#     return Image.fromarray(np_arr)
    
def get_name_from_path(path : str) -> str:
    """
        Function to get the <name> of a model from a path. Because the name is always in front of weights/best.pt
    """
    return os.path.normpath(path).split(os.sep)[-3]


def param_dict_from_name(filename : str, separator : str = "-") -> dict:
    """
        Function to get a parameter dictionary from a name
    """
    model_name  = get_name_from_path(filename)
    mn = model_name.split(separator)
    return {
        "date": mn[0],
        "model_size" : mn[1],
        "train_dataset" : mn[2],
        "test_dataset" : mn[3]
    }

def get_model_params(args) -> Tuple[dict, str]:
    """
        Function to get model name and parameter
    """
    # Test time
    param_dict = param_dict_from_name(args.model)
    param_dict["test_dataset"] = args.dataset
    model_name = "-".join(list(param_dict.values()))
    return param_dict, model_name