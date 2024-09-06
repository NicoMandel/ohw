import numpy as np
# import rawpy
from PIL import Image
import cv2
from ultralytics.utils.ops import xywhn2xyxy

def load_image(imgf: str):
    im = cv2.imread(imgf)
    rgb_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb_img

def load_label(label_f : str):
    return np.genfromtxt(label_f, delimiter=' ')

def det_to_bb(shape : tuple, detections : np.ndarray) -> list:
    """
        centroid x, centroid y, bb width, bb height
        correct covnersion: https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291 
        https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format 
        or this for simplicity: https://docs.ultralytics.com/usage/simple-utilities/#get-bounding-box-dimensions
        or even better, this one: https://docs.ultralytics.com/usage/simple-utilities/#bounding-boxes
        look through all of this for simplicity!
    """
    img_w, img_h = shape[:2]
    det_xyxy = xywhn2xyxy(detections[:,1:], img_w, img_h)
    det = np.c_[detections[:,0], det_xyxy]
    return det



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
