import os.path
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from ohw.dataset import DisplayDataset
from ohw.utils import get_site_dirs, param_dict_from_name, save_label

def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("model", type=str, help="Model Path. Will load a model and do predictions.")
    parser.add_argument("-o", "--output", action="store_true", help="Boolean value. If given, will create output folder structure with <visualisations> subfolder in subdirectory.")
    parser.add_argument("-l", "--label", action="store_true", help="Boolean value. If given, will also output a .txt label file for the folder in the yolo-specific format.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Boolean value. If given, will look recursively for subfolders <images> and <labels> and add them to the set.")
    return parser.parse_args()

def vis_prediction(input_dir : str, output : bool, model_p : str, label : bool = False):
    model = YOLO(model_p)
    model_params = param_dict_from_name(model_p)
    model_name = "-".join(list(model_params.values()))
    site_dir = os.path.abspath(input_dir)
    visdir = Path(os.path.abspath(os.path.join(site_dir, model_name)))
    if output or label:
        visdir.mkdir(exist_ok=True)
        print("Created: {}\nWill write images to it, if they don't exist yet".format(visdir))
    
    ds = DisplayDataset(input_dir)

    for ds_item in tqdm(ds, leave=True):
        img = ds_item[0]
        img_n = ds_item[1]
        result = model(img)[0]
        outpath = visdir / (img_n + ".png")
        result.save(outpath) if output else result.show()
        print("Test debug line")
        if label and result.boxes:
            bxs = result.boxes
            classes = bxs.cls.detach().cpu().numpy()
            conf =  bxs.conf.detach().cpu().numpy()
            xywhn = bxs.xywhn.detach().cpu().numpy()
            print("test debug line - see if we can output boxes")
            labels = np.c_[classes, xywhn]
            labelf = visdir / (img_n + ".txt")
            save_label(labels, labelf)
            print("saved: {}".format(labelf))
        # if output:
        #     result.save("sometest.jpg")

if __name__=="__main__":
    args = parse_args()
    if args.recursive:
        site_dirs = get_site_dirs(args.input)
        for site in site_dirs:
            vis_prediction(str(site), args.output, args.model, args.label)
    else:
        vis_prediction(args.input, args.output, args.model, args.label)