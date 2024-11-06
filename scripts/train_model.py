import os.path
from datetime import datetime
from argparse import ArgumentParser
from ultralytics import YOLO, settings
from test_model import test_model, find_conf
from ohw.utils import append_to_xlsx

def parse_args():
    parser = ArgumentParser(description="Script for training a model on a specified dataset.")
    parser.add_argument("size", type=str, help="Size of the model. Choose from n,s,m,l,x etc. Or a file")
    parser.add_argument("dataset", help="name of the .yaml file in the <data> folder.")
    parser.add_argument("-t", "--test", type=str, default=None, help="Test dataset, if different from train dataset file")
    parser.add_argument("-s", "--save", type=str, default=None, help="File where to store results")
    return parser.parse_args()

# https://docs.ultralytics.com/modes/train/#usage-examples
def train_model(model : YOLO, data_path, project : str, name : str):
    results = model.train(
        # model = os.path.join(basedir, "yolov8s.pt"),
        # model = "yolov8s.pt",
        data = data_path,
        epochs=300,
        imgsz=1280,
        batch=2,
        cache=False,
        # naming settings
        project=project,
        name=name
    )
    return results, model

if __name__=="__main__":
    args = parse_args()
    # Dataset
    settings.update({"datasets_dir" : "datasets"})
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    
    train_dataset = args.dataset
    test_dataset = args.test if args.test else train_dataset
    train_data_path = os.path.join(basedir, "data", "{}.yaml".format(train_dataset))       # https://github.com/ultralytics/ultralytics/issues/8823
    test_data_path = os.path.join(basedir, "data", "{}.yaml".format(test_dataset))

    # Model
    model_size = args.size
    model = YOLO("yolov8{}.pt".format(model_size))
    
    # Training time:
    today = datetime.today().strftime("%Y%m%d")
    param_dict = {
        "date": today,
        "model_size" : model_size,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }
    model_name = "-".join(list(param_dict.values()))
    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    results, model = train_model(model, train_data_path, project="results", name=model_name)
    metrics, model = test_model(model, test_data_path, project="results", name=model_name)
    confidence = find_conf(metrics)
    
    if args.save:
        out_dict = {**param_dict, **metrics.results_dict}
        out_dict["Confidence"] = confidence
        xlsxf = os.path.join(basedir, args.save)
        append_to_xlsx(model_name, out_dict, xlsxf)
    else:
        print("Model Results for model: {}".format(model_name))
        print(metrics.results_dict)