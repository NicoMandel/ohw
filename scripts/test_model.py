import os.path
from ultralytics import YOLO, settings

def test_model(model : YOLO, data_path,  project : str, name : str):
    metrics = model.val(
        data=data_path,
        imgsz=1280,
        split="test",
        # Naming conventions
        project=project,
        name=name
        )

    return metrics, model

if __name__=="__main__":
    settings.update({"datasets_dir" : "datasets"})
    model_size = "s"
    model = YOLO("yolov8{}.pt".format(model_size))
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    data_path = os.path.join(basedir, "data", "1cm.yaml")       # https://github.com/ultralytics/ultralytics/issues/8823

    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    results, model = test_model(model, data_path, project="results", name="1cm{}".format(model_size))
    test_model(model)
    print("test debug line")