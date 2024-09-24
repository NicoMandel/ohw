import os.path
from datetime import datetime
from ultralytics import YOLO, settings
from test_model import test_model
from ohw.utils import append_to_xlsx

# https://docs.ultralytics.com/modes/train/#usage-examples
def train_model(model : YOLO, data_path, project : str, name : str):
    results = model.train(
        # model = os.path.join(basedir, "yolov8s.pt"),
        # model = "yolov8s.pt",
        data = data_path,
        epochs=2,
        imgsz=1280,
        batch=2,
        cache=False,
        # naming settings
        project=project,
        name=name
    )
    return results, model

if __name__=="__main__":
    # Dataset
    settings.update({"datasets_dir" : "datasets"})
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    
    train_dataset = "1cm"
    test_dataset = train_dataset
    train_data_path = os.path.join(basedir, "data", "{}.yaml".format(train_dataset))       # https://github.com/ultralytics/ultralytics/issues/8823
    test_data_path = train_data_path

    # Model
    model_size = "s"
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
    

    out_dict = {**param_dict, **metrics.results_dict}
    xlsxf = os.path.join(basedir, "results", "yolov8.xlsx")
    append_to_xlsx(model_name, out_dict, xlsxf)