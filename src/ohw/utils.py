from typing import Tuple
import os.path
import subprocess
from pathlib import Path
import simplekml
from PIL import Image
import numpy as np
import torch
import pandas as pd
import rawpy
import matplotlib.pyplot as plt
import cv2
from ultralytics.utils.ops import xywhn2xyxy

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

def write_summary(summarydict : dict, outf : str, orig_len : int):
    """
        to write summary statistics to a file
    """
    with open(outf, 'w') as f:
        f.write("{} of {} files with predicted objects\n".format(len(summarydict), orig_len))
        [f.write("{}: {} detections\n".format(k, v)) for k, v in summarydict.items()]

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

# Geotagging images
def geotag_image(img_path : np.ndarray, geotag : pd.Series) :
    """
        Function to geotag an image with the exif tool   
    """
    # metad = _format_gps_metadata(geotag)
    lat = geotag['latitude [decimal degrees]']
    long = geotag['longitude [decimal degrees]']
    command=[
        "exiftool",
        f"-GPSLatitude={lat}",
        f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-GPSLongitude={long}",
        f"-GPSLongitudeRef={'E' if long >= 0 else 'W'}",
        "-overwrite_original",
        img_path
    ]
    subprocess.run(command)

# creating kmls
def create_kml(img_f : str, geodata : pd.Series, save_dir: str, img_id : str, compress : bool = True):
    """
        Function to create a kml of an image with the geolocation and embed the information
    """
    kml = simplekml.Kml()
    photo = kml.newphotooverlay(name=img_id)
    photo.camera = simplekml.Camera(longitude = geodata['longitude [decimal degrees]'], latitude=geodata['latitude [decimal degrees]'], altitude=geodata["altitude [meter]"],
                                    altitudemode=simplekml.AltitudeMode.clamptoground)
    photo.point.coords= [(geodata["longitude [decimal degrees]"], geodata["latitude [decimal degrees]"])]
    if compress:
        img_f = _compress_img(img_f, save_dir, img_id)
    photo.icon.href = "{}".format(img_f)
    photo.viewvolume = simplekml.ViewVolume(-25,25,-15,15,1)
    kml_path = os.path.join(save_dir, "{}.kmz".format(img_id))
    kml.savekmz(kml_path)
    # remove file after compressing into kmz
    if compress:
        os.remove(img_f)
        # pass

def _compress_img(img_f : str, save_dir, img_id, max_size : int = 512) -> str:
    """
        storing a thumbnail of an image.
        Returns the str where it is returned
    """
    img = Image.open(img_f).copy()
    img.thumbnail((max_size, max_size))
    thumb_path = os.path.join(save_dir, "{}_thumb.png".format(img_id))
    img.save(thumb_path)
    return thumb_path


# testing GPU access
def test_gpu():
    print("Cuda is available {}".format(torch.cuda.is_available()))
    print("Device Count: {}".format(torch.cuda.device_count()))
    cd = torch.cuda.current_device()
    print("Current device: {}".format(cd))
    print("Device name: {}".format(torch.cuda.get_device_name(cd)))
    print("Device properties: {}".format(torch.cuda.get_device_properties(cd)))
    