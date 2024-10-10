import os.path
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from ohw.dataset import DisplayDataset, DisplayLabelsDataset
from ohw.utils import det_to_bb, save_image, get_site_dirs, param_dict_from_name

def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("model", type=str, help="Model Path. Will load a model and do predictions.")
    parser.add_argument("-l", "--label", action="store_true", help="Boolean value. If set, will attempt to load labels and display images side by side (in addition to writing to drive)")
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure with <visualisations> subfolder in subdirectory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Boolean value. If given, will look recursively for subfolders <images> and <labels> and add them to the set.")
    return parser.parse_args()

def vis_prediction(input_dir : str, output : bool, model_p : str, label : bool = False):
    model = YOLO(model_p)
    model_params = param_dict_from_name(model_p)
    model_name = "-".join(list(model_params.values()))
    site_dir = os.path.abspath(input_dir)
    visdir = Path(os.path.abspath(os.path.join(site_dir, model_name)))
    if output:
        visdir.mkdir(exist_ok=True)
        print("Created: {}\nWill write images to it, if they don't exist yet".format(visdir))
    
    ds = DisplayLabelsDataset(input_dir) if label else DisplayDataset(input_dir)

    for ds_item in tqdm(ds, leave=True):
        img = ds_item[0]
        img_n = ds_item[1]
        result = model(img)[0]
        # result.show()
        if result.boxes:
            ann = Annotator(
                img,
                line_width=None,  # default auto-size
                font_size=None,  # default auto-size
                font="Arial.ttf",  # must be ImageFont compatible
                pil=False,  # use PIL, otherwise uses OpenCV
            )
            for box in result.boxes.xyxy.cpu():
                ann.box_label(box, label="OHW")
            img_w_bboxes = ann.result()
            plt.imshow(img_w_bboxes)
            plt.show()
            
        if label:
            label = ds_item[2]


if __name__=="__main__":
    args = parse_args()
    if args.recursive:
        site_dirs = get_site_dirs(args.input)
        for site in site_dirs:
            vis_prediction(str(site), args.output, args.model, args.label)
    else:
        vis_prediction(args.input, args.output, args.model, args.label)