import os.path
from ultralytics import YOLO, settings

# https://docs.ultralytics.com/modes/train/#usage-examples

if __name__=="__main__":
    settings.update({"datasets_dir" : "datasets"})
    model = YOLO("yolov8s.pt")
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    data_path = os.path.join(basedir, "data", "1cm.yaml")       # https://github.com/ultralytics/ultralytics/issues/8823

    # training settings: https://docs.ultralytics.com/modes/train/#train-settings
    results = model.train(
        # model = os.path.join(basedir, "yolov8s.pt"),
        # model = "yolov8s.pt",
        data = data_path,
        epochs=100,
        imgsz=1280,
        batch=2,
        cache=False,
        # naming settings
        project="results",
        name="1cm"
        )
    print("test debug line")