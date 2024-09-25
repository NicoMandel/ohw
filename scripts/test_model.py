import os.path
from argparse import ArgumentParser
from datetime import datetime
from ultralytics import YOLO, settings
from ohw.utils import append_to_xlsx, replace_test_dataset

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

if __name__=="__main__":
    args = parse_args()
    settings.update({"datasets_dir" : "datasets"})
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    test_dataset = args.dataset
    data_path = os.path.join(basedir, "data", "{}.yaml".format(test_dataset))       # https://github.com/ultralytics/ultralytics/issues/8823

    # Model settings
    model = YOLO(args.model)

    # Test time
    today = datetime.today().strftime("%Y%m%d")
    param_dict = {
        "date": today,
        "train_dataset": "NA",
        "test_dataset": test_dataset
    }
    model_name = args.name if args.name else replace_test_dataset(args.model, test_dataset)
    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    metrics, model = test_model(model, data_path, project="results", name=model_name)
    
    if args.save:
        out_dict = {**param_dict, **metrics.results_dict}
        xlsxf = os.path.join(basedir, args.save)
        append_to_xlsx(model_name, out_dict, xlsxf)
    else:
        print("Model Results for model: {}".format(model_name))
        print(metrics.results_dict)
