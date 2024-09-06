# https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.autosplit
# auto-labelling-> https://docs.ultralytics.com/usage/simple-utilities/?h=sam+labeling

import os.path
from ultralytics.data.utils import autosplit


if __name__=="__main__":
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))

    ds_name = "0.24cm"
    dataset_path = os.path.join(basedir, "datasets", ds_name, "train")

    autosplit(
        path=dataset_path,
        weights=(0.8, 0.2, 0.0),
        annotated_only=False
    )

    # print("Test Debug line")
    
