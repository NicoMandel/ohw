import os.path
from pathlib import Path
import glob
from PIL import Image
import torch
import numpy as np

from torchvision.datasets.vision import VisionDataset
from ultralytics.data.utils import IMG_FORMATS, FORMATS_HELP_MSG, img2label_paths
from ohw.utils import load_image, load_label

class DisplayLabelsDataset(VisionDataset):
    def __init__(self, root : str, img_dir = "images", ldir = "labels") -> None:
        super().__init__(root=root)

        # default
        self.root = os.path.abspath(root)
        self.imgdir = Path(self.root) / img_dir
        self.labeldir = Path(self.root) / ldir
        
        # 
        self.img_list = self.get_img_files(self.imgdir)
        self.label_list = self.get_label_files()

    def __len__(self) -> int:
        return len(self.img_list)
    
    def __getitem__(self, index: int) -> tuple[Image.Any, ]:
        imgf = self.img_list[index]
        img_id = Path(imgf).stem
        lid = img_id + ".txt"
        lf = os.path.join(self.labeldir, lid)

        img = load_image(imgf)
        labelarr = load_label(lf)
        return img, labelarr, img_id

    # from yolov8 basedataset
    def get_img_files(self, img_path):
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n") from e
        return im_files
    
    def get_label_files(self):
        self.label_files = img2label_paths(self.img_list)