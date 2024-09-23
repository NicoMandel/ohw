import os.path
from ultralytics import YOLO, settings
from test_model import test_model

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
    settings.update({"datasets_dir" : "datasets"})
    model_size = "s"
    model = YOLO("yolov8{}.pt".format(model_size))
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    data_path = os.path.join(basedir, "data", "1cm.yaml")       # https://github.com/ultralytics/ultralytics/issues/8823

    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    results, model = train_model(model, data_path, project="results", name="1cm{}".format(model_size))
    metrics, model = test_model(model, data_path, project="results", name="1cm{}".format(model_size))
    print("test debug line")