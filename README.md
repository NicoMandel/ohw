# ohw

## usage:
Run with: `docker run -it --gpus all yolov8`

## Troubleshooting
If script does not launch or outputs `Cuda is available: False`, then try [this for accessing GPUs](https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours)


## Dumping files
### Conda
dump conda files with `conda env export --from-history > environment.yaml` will only export the packages specifically requested to install and function cross-plattform

### pip 
use `pipreqs .` to dump 