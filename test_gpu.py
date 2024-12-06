import sys
import torch
import ultralytics
import sahi
import tqdm
import PIL
import simplekml
import exiftool

if __name__=="__main__":
    print("Python: {}".format(sys.version))
    print("Cuda:{}".format(torch.cuda.is_available()))
    print("Cuda Version: {}".format(torch.version.cuda))
    print("Ultraltics: {}".format(ultralytics.__version__))