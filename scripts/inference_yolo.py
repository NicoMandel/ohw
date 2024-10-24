import os.path
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from ohw.dataset import DisplayDataset
from ohw.utils import get_site_dirs, param_dict_from_name, save_label, write_summary

def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input image folder")
    parser.add_argument("model", type=str, help="Model Path. Will load a model and do predictions.")
    parser.add_argument("output", type=str, help="Output location for the label files. Will create <model_name>/<labels> subfolder here.")
    parser.add_argument("-s", "--summary", action="store_true", help="If given, will create summary statistics - file with count of instances per image and list of images with detections.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Boolean value. If given, will look recursively for subfolders <images> and <labels> and add them to the set.")
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the dataset to be used inside the <model_name>/dataset/<labels> folder. If none given, will use folder name of ds")
    return parser.parse_args()

def save_prediction(input_dir : str, output : str, model_p : str, summary : bool, name : str = None):  # visualise : bool = False
    # model loading and name
    model = YOLO(model_p)
    model_params = param_dict_from_name(model_p)
    model_name = "-".join(list(model_params.values()))
    
    # output directory creation
    site_dir = os.path.abspath(input_dir)
    ds_name = name if name else os.path.basename(os.path.normpath(site_dir))
    outdir_p = os.path.abspath(os.path.join(output, model_name, ds_name))
    outdir_l = Path(os.path.join(outdir_p, "labels"))
    outdir_l.mkdir(exist_ok=True, parents=True)
    print("Created: {}\nWill write labels to it".format(outdir_l))
    
    # dataset for only loading images
    ds = DisplayDataset(site_dir, img_dir=None)
    summary = {}
    for ds_item in tqdm(ds, leave=True):
        img = ds_item[0]
        img_n = ds_item[1]
        result = model(img, verbose=False)[0]
        
        # only write output if there is something to output
        if result.boxes:
            bxs = result.boxes
            classes = bxs.cls.detach().cpu().numpy()
            conf =  bxs.conf.detach().cpu().numpy()
            xywhn = bxs.xywhn.detach().cpu().numpy()
            labels = np.c_[classes, xywhn]
            labelf = outdir_l / (img_n + ".txt")
            save_label(labels, labelf)
            
            # get summary statistics
            summary[img_n] = labels.shape[0]
    print("Saved {} files with detections from {} of original dataset to {}".format(len(summary), len(ds), outdir_l))    
    if summary:
        summaryf = os.path.join(outdir_p, "{}_summary.txt".format(model_name))
        write_summary(summary, summaryf, orig_len = len(ds))
        
        
        
if __name__=="__main__":
    args = parse_args()
    if args.recursive:
        site_dirs = get_site_dirs(args.input)
        for site in site_dirs:
            save_prediction(str(site), args.output, args.model, args.summary)
    else:
        save_prediction(args.input, args.output, args.model, args.summary, args.name)