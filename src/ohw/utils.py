from typing import Tuple
import os
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import rawpy
import psutil
import matplotlib.pyplot as plt
import cv2
from ultralytics.utils.ops import xywhn2xyxy

# debugging memory usage
def print_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    print(f"[{stage}] Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")


def log_memory_usage(i : int = 0):
    mem_usage = mem_usage_of_process_tree()
    print(f"Memory RSS usage at end of iteration {i} including subprocesses: {mem_usage:.2f} MB")  
    print(f"Virtual memory percentage used: {psutil.virtual_memory().percent}")

def mem_usage_of_process_tree(pid=None):
    if pid is None:
        pid = os.getpid()
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)  
        total_mem = sum(p.memory_info().rss for p in processes) # RSS = Resident Set Size (physical memory)
        return total_mem / 1024 ** 2 # in mb
    except psutil.NoSuchProcess:
        return 0.0 # terminated process

def load_image(imgf: str, convert : bool = False) -> np.ndarray:
    if imgf.lower().endswith(".arw"):
        im = load_arw(imgf)
    else:
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

def load_arw(fpath : str, cvtcolor : bool = True) -> np.ndarray:
    # raw = rawpy.imread(fpath)
    # img= raw.postprocess(use_camera_wb=True, output_bps=8)
    with rawpy.imread(fpath) as raw:
        img=raw.postprocess(use_camera_wb=True, output_bps=8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if cvtcolor else img
    return img

def get_dataset_name_from_path(yamlpath : str) -> str:
    return Path(yamlpath).stem

def get_size_from_filename(modelfile : str) -> str:
    paramd = param_dict_from_name(modelfile)
    return paramd["model_size"]

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
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(l[0].astype(int), l[1], l[2], l[3], l[4]))
    return None

def write_summary(summarydict : dict, outf : str, orig_len : int):
    """
        to write summary statistics to a file
    """
    with open(outf, 'w') as f:
        f.write("{} of {} files with predicted objects\n".format(len(summarydict), orig_len))
        [f.write("{}: {} detections\n".format(k, v)) for k, v in summarydict.items()]
    return None

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

def convert_bbox_coco2yolo(img_w : int, img_h : int, xywh_arr : np.ndarray) -> np.ndarray:
    """
        https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco
    """
    xyc_wh = np.c_[ (xywh_arr[:,0] + (xywh_arr[:,2] / 2)) / img_w,
                    (xywh_arr[:,1] + (xywh_arr[:,3] /2)) / img_h,
                    xywh_arr[:,2] / img_w,
                    xywh_arr[:,3] / img_h
                    ]
    # xyxy = xywh2xyxy(xyc_wh)
    # xywhn = xyxy2xywhn(xyxy, img_w, img_h)
    return xyc_wh

def convert_pred(pred):
    img_w = pred.image_width
    img_h = pred.image_height
    xywh_arr = np.asarray([bbox.to_coco_annotation().bbox for bbox in pred.object_prediction_list])
    cats = np.array([bbox.category.id for bbox in pred.object_prediction_list])   # bbox.score.value,
    xywhn_arr = convert_bbox_coco2yolo(img_w, img_h, xywh_arr)
    yolo_bboxes = np.c_[cats, xywhn_arr]
    
    # if not yolo_bboxes: yolo_bboxes = None
    return yolo_bboxes

def plot_coco(img : np.ndarray, results):
    img_w_annot = np.copy(img)
    for bbox in results.object_prediction_list:
        cc_bb = bbox.to_coco_annotation().bbox
        x1 = int(np.floor(cc_bb[0]))
        y1 = int(np.floor(cc_bb[1]))
        x2 = int(np.ceil(x1 + cc_bb[2]))
        y2 = int(np.ceil(y1 + cc_bb[3]))
        cv2.rectangle(img_w_annot, (x1, y1), (x2, y2), color=(255, 0,0), thickness=2)

    plt.imshow(img_w_annot)
    plt.show()

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
    img_h, img_w = shape[:2]
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
            df_prev = pd.read_excel(open(xlsx_f, 'rb'), header=0, index_col=0)
            result = pd.concat([df_prev, df])
        else:
            result = df
        with pd.ExcelWriter(xlsx_f) as writer:
            result.to_excel(writer)
        return True
    except OSError as ose:
        print(ose)
        return False

def get_model(path : str, resolution : str, confidence : float, metric : str = 'fitness') -> tuple:
    if path.endswith(".xlsx"):
        print("Loading best model for resolution {} from registry file at: {}".format(resolution, path))
        return get_model_from_xlsx(path, resolution, metric)
    else:
        model_name = os.path.basename(os.path.normpath(path))
        print("Using model {} with pre-specified confidence {}. No check on resolution here!".format(model_name, confidence))
        return model_name, confidence

def get_model_from_xlsx(xlsx_path : str, resolution : str, metric : str = 'fitness') -> tuple:
    """
        function to get confidence and best model nam for a specific resolution from a path.
        Resolution should be a string with either 1cm or 024cm in it
    """
    df = pd.read_excel(open(xlsx_path, 'rb'), header=0, index_col=0)
    subdf = df[df['test_dataset'] == resolution]
    max_id = subdf[metric].idxmax()
    confidence = subdf.loc[max_id, "Confidence"]
    return max_id, confidence

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
    p = Path(path).resolve()
    return os.path.normpath(str(p)).split(os.sep)[-3]


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


# testing GPU access
def test_gpu():
    print("Cuda is available {}".format(torch.cuda.is_available()))
    print("Device Count: {}".format(torch.cuda.device_count()))
    cd = torch.cuda.current_device()
    print("Current device: {}".format(cd))
    print("Device name: {}".format(torch.cuda.get_device_name(cd)))
    print("Device properties: {}".format(torch.cuda.get_device_properties(cd)))
    

def _check_bbox_dim(box: np.ndarray, shape, minval) -> tuple[tuple, tuple]:
    """
        box (tuple): The bounding box coordinates (x1, y1, x2, y2).
        img_h = x
        img_w = y
        Function to check whether bounding boxes are appropriately dimensioned and re-dimension if necessary
    """
    img_y, img_x = shape[:2]
    tl = box[2:]
    br = box[:2]
    # inflate to minimum if necessary. Check if the dims are outside of the image, if yes, shift the bbox entirely by that.
    # x-dimesion, width
    if (br[1] - tl[1]) < minval:
        ctr = __calc_centroid(br[1], tl[1])
        l, r = int(ctr - minval / 2), int(ctr + minval / 2)
        if l < 0:
            r += 0-l
            l=0
        if r > img_y:
            l -= r-img_y
            r = img_y
    else:
        l = tl[1]
        r = br[1]
    
    # y - dimension. height
    if (br[0] - tl[0]) < minval:
        ctr = __calc_centroid(br[0], tl[0])
        t, b = int(ctr - minval / 2), int(ctr + minval / 2)
        if t < 0:
            b += 0-t
            t = 0
        if b > img_x:
            t -= b-img_x
            b = img_x
    else:
        b = br[0]
        t = tl[0]
    ntl = (t, l)
    nbr = (b, r)
    return ntl, nbr

def __calc_centroid(high : tuple, low: tuple) -> tuple:
    """
        Function to calculate the center pixel. can be applied to both dimensions
    """
    return int((low + high) /2)

def annotate_image_w_buffer(img : np.ndarray, detections : np.ndarray, rgb : tuple = (0, 0, 255), line_width : int = None, minval : int =10) -> np.ndarray:
        """
            function from ultralytics in: plotting.py
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        """
        det_xyxy = det_to_bb(img.shape, detections).astype(np.int32)
        line_width = 5 if line_width is None else line_width
        for row in det_xyxy:
            c_idx, *box = row
            tl, br = _check_bbox_dim(box, img.shape, minval)
            cv2.rectangle(img, tl, br, rgb , line_width)
        return img