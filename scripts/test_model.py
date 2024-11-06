import os.path
from argparse import ArgumentParser
from ultralytics import YOLO, settings
import numpy as np
from ohw.utils import append_to_xlsx, get_model_params
from ultralytics.utils.metrics import DetMetrics

def parse_args():
    parser = ArgumentParser(description="Script for testing a model on a specified dataset.")
    parser.add_argument("model", type=str, help="Model file to test")
    parser.add_argument("dataset", help="name of the .yaml file in the <data> folder.")
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the model to be used in storing the file")
    parser.add_argument("-s", "--save", type=str, default=None, help="File where to store results. If None given, will print")
    return parser.parse_args()

def test_model(model : YOLO, data_path, project : str = None, name : str = None):
    metrics = model.val(
        data=data_path,
        imgsz=1280,
        split="test",
        plots=True,
        # # Naming conventions
        project=project,
        name=name 
        )
    return metrics, model

def find_conf(metrics : DetMetrics):
    """
        Function to find the confidence where the F1 score is the highest on the test dataset. 
        Ballpark and underestimated. See: https://github.com/ultralytics/ultralytics/blob/f2a7a29e531ad029255c8ec180ff65de24f42e8d/ultralytics/yolo/utils/metrics.py#L406
    """
    f1c_id = metrics.curves.index('F1-Confidence(B)')
    f1c = metrics.curves_results[f1c_id]
    f1_max = np.argmax(f1c[1])
    conf = f1c[0][f1_max-1]
    return conf

if __name__=="__main__":
    args = parse_args()
    settings.update({"datasets_dir" : "datasets"})
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    test_dataset = args.dataset
    data_path = os.path.join(basedir, "data", "{}.yaml".format(test_dataset))       # https://github.com/ultralytics/ultralytics/issues/8823

    # Model settings
    model = YOLO(args.model)
    param_dict, model_name = get_model_params(args)

    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    metrics, model = test_model(model, data_path, project="results", name=model_name)
    confidence = find_conf(metrics)

    if args.save:
        out_dict = {**param_dict, **metrics.results_dict}
        # also add the confidence on the test dataset to the metrics.
        out_dict["Confidence"] = confidence
        xlsxf = os.path.join(basedir, args.save)
        append_to_xlsx(model_name, out_dict, xlsxf)
    else:
        print("Model Results for model: {}".format(model_name))
        print(metrics.results_dict)
