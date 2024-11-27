import os.path
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from display_dataset import annotate_image
from ohw.dataset import DisplayDataset
from ohw.utils import save_image, save_label, convert_pred, get_model_from_xlsx
from ohw.log_utils import log_exists, read_log, append_to_log, summary_exists, append_to_summary, create_summary, postprocess_summary

# https://docs.ultralytics.com/guides/sahi-tiled-inference/#batch-prediction
def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("registry", type=str, help="Path of registry xlsx. Will choose best model for given resolution from this, with the metric specified.")
    parser.add_argument("resolution", type=str, help="Resolution which model should be loaded. Choose from 024cm and 1cm.")
    parser.add_argument("output", type=str, help="Output location for the label files. Will create <model_name>/<labels> subfolders here.")
    
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the dataset to be used inside the <model_name>/dataset/<labels> folder. If none given, will use folder name of ds")
    parser.add_argument("-s", "--summary", action="store_true", help="If given, will create summary statistics - file with count of instances per image and list of images with detections.")
    parser.add_argument("-v", "--visualise", action="store_true", help="If true, will also write <model_name>/dataset/<visualisations> folder.")
    parser.add_argument("-m", "--metric", type=str, default="fitness", help="Metric that is used to select the maximum model from the registry. Defaults to fitness")

    parser.add_argument("--ratio", default=0.3, type=float, help="Overlap ratio, vertical as well as horizontal. Default 0.3")
    parser.add_argument("--size", default=1280, type=int, help="Model size to be used for inference. Defaults to 1280.")
    return parser.parse_args()

def sahi(input_dir : str, registry_f : str, resolution : str, output : str, name : str, summary : bool, visualise : bool, metric : str, overlap : float = 0.3, model_size : int = 1280):
    # Model
    model_name, conf_thresh = get_model_from_xlsx(registry_f, resolution, metric)
    model_p = os.path.join(os.path.dirname(registry_f) , model_name, 'weights', 'best.pt')
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_p,
        confidence_threshold=conf_thresh,
        device="cuda:0",  # or 'cpu'
    )
    
    # Dataset
    site_dir = os.path.abspath(input_dir)
    ds_name = name if name else os.path.basename(os.path.normpath(site_dir))
    outdir_p = os.path.abspath(os.path.join(output, model_name, ds_name))
    outdir_l = Path(os.path.join(outdir_p, "labels"))
    outdir_l.mkdir(exist_ok=True, parents=True)
    print("Created: {}\nWill write labels to it".format(outdir_l))
    
    if visualise:
        outdir_v = Path(os.path.join(outdir_p, "visualisations"))
        outdir_v.mkdir(exist_ok=True)
        print("Created: {}\nWill write images to it".format(outdir_v))

    # logging
    log_list = []
    if log_exists(outdir_p):
        log_list = read_log(outdir_p)
    
    # Setup model
    ds = DisplayDataset(input_dir, img_dir=None)
    
    # if summary file doesn't exist write first line - which model processes how many images
    if not summary_exists(outdir_p, model_name): create_summary(outdir_p, model_name, len(ds))

    print("Performing inference on a total of {} images. Checking if already processed first".format(len(ds)))
    for ds_item in tqdm(ds):
        img = ds_item[0]
        img_n = ds_item[1]
        # continue, if already in log
        if img_n in log_list:
            print("Image {} already in log. Skipping.".format(img_n))
            continue
        # save_image(img, "sometest.png") # ! this resulted that it only worked with conversion
        result = get_sliced_prediction(
            Image.fromarray(img, mode="RGB"),
            detection_model,
            slice_height=model_size,
            slice_width=model_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap
            )

        if result.object_prediction_list:
            labels = convert_pred(result)
            labelf = outdir_l / (img_n + ".txt")
            save_label(labels, labelf)
            # if visualisations are asked for, output them too
            # if visualise:
            img_w_bboxes = annotate_image(img, labels, line_width=2, font_size=6)
            visf = outdir_v / (img_n + ".jpg") 
            save_image(img_w_bboxes, str(visf), cvtcolor=True)
            
            # if summary given, append to summary file
            if summary:
                append_to_summary(outdir_p, model_name, img_n, labels.shape[0])
        
        append_to_log(outdir_p, img_n)
    if summary:
        postprocess_summary(outdir_p, model_name, len(ds))
    
        
if __name__=="__main__":
    args = parse_args()
    sahi(args.input, args.registry, args.resolution, args.output, args.name, args.summary, args.visualise, args.metric, args.ratio, args.size)