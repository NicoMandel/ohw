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
from ohw.utils import param_dict_from_name, save_image, save_label, write_summary, convert_pred, plot_coco

# https://docs.ultralytics.com/guides/sahi-tiled-inference/#batch-prediction
def parse_args():
    parser = ArgumentParser(description="Script for displaying the images for a site with associated labels.")
    parser.add_argument("input", type=str, help="Location of the input folder, as root. Will find <images> and <labels> subfolders.")
    parser.add_argument("model", type=str, help="Model Path. Will load a model and do predictions.")
    parser.add_argument("output", type=str, help="Output location for the label files. Will create <model_name>/<labels> subfolders here.")
    
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the dataset to be used inside the <model_name>/dataset/<labels> folder. If none given, will use folder name of ds")
    parser.add_argument("-s", "--summary", action="store_true", help="If given, will create summary statistics - file with count of instances per image and list of images with detections.")
    parser.add_argument("-v", "--visualise", action="store_true", help="If true, will also write <model_name>/dataset/<visualisations> folder.")
    
    parser.add_argument("--confidence", default=0.11, type=float, help="Cutoff Confidence, under which detections will be discarded. Default 0.11")
    parser.add_argument("--ratio", default=0.3, type=float, help="Overlap ratio, vertical as well as horizontal. Default 0.3")
    parser.add_argument("--size", default=1280, type=int, help="Model size to be used for inference. Defaults to 1280.")
    return parser.parse_args()

def sahi(input_dir : str, model_p : str, output : str, name : str, summary : bool, visualise : bool, conf_thresh: float = 0.5, overlap : float = 0.3, model_size : int = 1280):
    # Model
    model_params = param_dict_from_name(model_p)
    model_name = "-".join(list(model_params.values()))
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

    # Setup model
    ds = DisplayDataset(input_dir, img_dir=None)
    summary = {}
    print("Performing inference on {} images".format(len(ds)))
    for ds_item in tqdm(ds):
        img = ds_item[0]
        img_n = ds_item[1]
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
            summary[img_n] = labels.shape[0]
            # if visualisations are asked for, output them too
            # if visualise:
            img_w_bboxes = annotate_image(img, labels, line_width=2, font_size=6)
            visf = outdir_v / (img_n + ".jpg") 
            save_image(img_w_bboxes, str(visf), cvtcolor=True)
            # print("Reading image: {}".format(img_n))
            # save_image(img_w_bboxes, "testimg.png")
            # plt.imshow(img_w_bboxes)
            # plt.show()
            # plot_coco(img, result)
            # get summary statistics
            summary[img_n] = labels.shape[0]
    print("Saved {} files with detections from {} of original dataset to {}".format(len(summary), len(ds), outdir_l))    
    if summary:
        summaryf = os.path.join(outdir_p, "{}_summary.txt".format(model_name))
        write_summary(summary, summaryf, orig_len = len(ds))
        
if __name__=="__main__":
    args = parse_args()
    sahi(args.input, args.model, args.output, args.name, args.summary, args.visualise, args.confidence, args.ratio, args.size)