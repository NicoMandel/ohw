import os
from datetime import datetime
from argparse import ArgumentParser
import torch

from ultralytics import YOLO, settings
from test_model import test_model, find_conf
from ohw.utils import append_to_xlsx, get_dataset_name_from_path, get_size_from_filename

os.environ["YOLO_VERBOSE"] = "false"
os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "-1"

def parse_args():
    parser = ArgumentParser(description="Script for training a model on a specified dataset.")
    parser.add_argument("model", type=str, help="Size of the model or path to model file. Choose from n,s,m,l,x etc. Or a file.")
    parser.add_argument("dataset", type=str, help="Absolute path of the data.yaml file")
    parser.add_argument("-t", "--test", type=str, default=None, help="Test dataset, if different from train dataset file.")
    parser.add_argument("-s", "--save", type=str, default=None, help="Absolute path to file where to store results.")
    parser.add_argument("-b", "--batch", type=int, default=2, help="Batch Size. defaults to 2")
    return parser.parse_args()

# https://docs.ultralytics.com/modes/train/#usage-examples
def train_model(model : YOLO, data_path, project : str, name : str, batch_size : int =2):
    results = model.train(
        # model = os.path.join(basedir, "yolov8s.pt"),
        # model = "yolov8s.pt",
        data = data_path,
        epochs=300,
        imgsz=1280,
        batch=batch_size,
        cache=False,
        # naming settings
        project=project,
        name=name,
        device=torch.cuda.current_device()
    )
    return results, model

if __name__=="__main__":
    args = parse_args()
    
    # Dataset
    settings.update({"datasets_dir" : "/home/ubuntu/"})
    dataset = args.dataset
    test_dataset = args.test if args.test else dataset
    # train_data_path = os.path.join(basedir, "data", "{}.yaml".format(train_dataset))       # https://github.com/ultralytics/ultralytics/issues/8823
    # test_data_path = os.path.join(basedir, "data", "{}.yaml".format(test_dataset))

    # Model
    modelf = args.model
    model = YOLO("yolov8{}.pt".format(modelf)) if len(modelf) == 1 else YOLO(modelf)
    
    # Training time:
    today = datetime.today().strftime("%Y%m%d")
    param_dict = {
        "date": today,
        "model_size" : modelf if len(modelf) == 1 else get_size_from_filename(modelf),
        "train_dataset": get_dataset_name_from_path(dataset),
        "test_dataset": get_dataset_name_from_path(test_dataset),
    }
    model_name = "-".join(list(param_dict.values()))
    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    results, model = train_model(model, dataset, project="results", name=model_name, batch_size=args.batch)
    metrics, model = test_model(model, test_dataset, project="results", name=model_name)
    confidence = find_conf(metrics)
    if args.save:
        out_dict = {**param_dict, **metrics.results_dict}
        out_dict["Confidence"] = confidence
        append_to_xlsx(model_name, out_dict, args.save)
    print("Model Results for model: {}".format(model_name))
    print(metrics.results_dict)
    print("Confidence at best F1 score: {}".format(confidence))