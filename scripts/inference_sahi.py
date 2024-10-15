import os.path
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ohw.dataset import DisplayDataset
from ohw.utils import param_dict_from_name, get_site_dirs, save_image

# https://docs.ultralytics.com/guides/sahi-tiled-inference/#batch-prediction
def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("model", type=str, help="Model Path. Will load a model and do predictions.")
    parser.add_argument("-l", "--label", action="store_true", help="Boolean value. If given, will also output a .txt label file for the folder in the yolo-specific format.")
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure with <visualisations> subfolder in subdirectory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Boolean value. If given, will look recursively for subfolders <images> and <labels> and add them to the set.")
    return parser.parse_args()

def sahi(input_dir : str, output : bool, model_p : str, conf_thresh: float = 0.5, label : bool = False):
    model_params = param_dict_from_name(model_p)
    model_name = "-".join(list(model_params.values()))
    site_dir = os.path.abspath(input_dir)
    visdir = Path(os.path.abspath(os.path.join(site_dir, model_name)))
    if output or label:
        visdir.mkdir(exist_ok=True)
        print("Created: {}\nWill write images to it, if they don't exist yet".format(visdir))

    # Setup model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_p,
        confidence_threshold=conf_thresh,
        device="cuda:0",  # or 'cpu'
    )
    ds = DisplayDataset(input_dir, img_dir=None)
    for ds_item in tqdm(ds, leave=True):
        img = ds_item[0]
        img_n = ds_item[1]
        # Prediction
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # save_image(img, "sometest.png") # ! this resulted that it only worked with conversion
        result = get_sliced_prediction(
            Image.fromarray(img),
            detection_model,
            slice_height=1280,
            slice_width=1280,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3
            )
        if output and result.object_prediction_list:
            result.export_visuals(export_dir=str(visdir))
            print("Written image: {} to: {}".format(img_n, visdir))

if __name__=="__main__":
    args = parse_args()
    if args.recursive:
        site_dirs = get_site_dirs(args.input)
        for site in site_dirs:
            sahi(str(site), args.output, args.model, args.label)
    else:
        sahi(args.input, args.output, args.model, args.label)