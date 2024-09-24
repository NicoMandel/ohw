import os.path
from datetime import datetime
from ultralytics import YOLO, settings
from ohw.utils import append_to_xlsx

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
    settings.update({"datasets_dir" : "datasets"})
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    test_dataset = "1cm"
    data_path = os.path.join(basedir, "data", "{}.yaml".format(test_dataset))       # https://github.com/ultralytics/ultralytics/issues/8823

    # Model settings
    model_size = "s"
    model = YOLO("yolov8{}.pt".format(model_size))

    # Test time
    today = datetime.today().strftime("%Y%m%d")
    param_dict = {
        "date": today,
        "model_size" : model_size,
        "train_dataset": "NA",
        "test_dataset": test_dataset
    }
    model_name = "-".join(list(param_dict.values()))
    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    metrics, model = test_model(model, data_path, project="results", name=model_name)
    
    
    out_dict = {**param_dict, **metrics.results_dict}
    xlsxf = os.path.join(basedir, "results", "yolov8.xlsx")
    append_to_xlsx(model_name, out_dict, xlsxf)
