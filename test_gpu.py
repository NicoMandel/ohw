import sys
import torch
import ultralytics

if __name__=="__main__":
    print("Python: {}".format(sys.version))
    print("Cuda:{}".format(torch.cuda.is_available()))
    print("Ultraltics: {}".format(ultralytics.__version__))