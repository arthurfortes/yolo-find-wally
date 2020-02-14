from darkflow.net.build import TFNet
import silence_tensorflow
import cv2


options = {
           "model": "cfg/yolov2-wally.cfg",
           "load": "cfg/yolov2-wally.weights",
           "batch": 8,
           "epoch": 5,
           "gpu": 0.8,
           "train": True,
           "annotation": "data/total/annotations/",
           "dataset": "data/total/images/",
           "load": -1
           }


tfnet = TFNet(options)
tfnet.train()
tfnet.savepb()
