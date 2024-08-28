from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# https://docs.ultralytics.com/guides/sahi-tiled-inference/#batch-prediction

if __name__=="__main__":
    # Download model and demo images
    yolov8_model_path = "models/yolov8s.pt"
    # download_yolov8s_model(yolov8_model_path)

    demo_ds_path = "datasets/demo"
    # download_from_url(
    #     "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    #     "{}/small-vehicles1.jpeg".format(demo_ds_path),
    # )
    # download_from_url(
    #     "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    #     "{}/terrain2.png".format(demo_ds_path),
    # )

    # Setup model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device="cuda:0",  # or 'cpu'
    )
    # Prediction
    result = get_sliced_prediction(
        "{}/small-vehicles1.jpeg".format(demo_ds_path),
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
        )
    result.export_visuals(export_dir=demo_ds_path)