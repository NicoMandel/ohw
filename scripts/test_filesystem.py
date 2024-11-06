import sys
import torch
import ultralytics
import os.path
import ohw.utils
import ohw.dataset

if __name__=="__main__":
    print("Imports of ohw subfolder appears to work")
    print("Python: {}".format(sys.version))
    print("Cuda:{}".format(torch.cuda.is_available()))
    print("Ultraltics: {}".format(ultralytics.__version__))