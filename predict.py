#!/usr/bin/env python
# coding: utf-8

from darkflow.net.build import TFNet
import tensorflow as tf
import numpy as np
import silence_tensorflow
import imutils
import cv2
import sys


options = {"pbLoad": "yolov2-wally.pb",
           "metaLoad": "yolov2-wally.meta"}
yolo_sensor = TFNet(options)


def highlight_wally(img, predictions, padding=20):
    predictions.sort(key=lambda x: x.get('confidence'))
    
    for pred in predictions:
        xtop = pred.get('topleft').get('x')
        ytop = pred.get('topleft').get('y')
        xbottom = pred.get('bottomright').get('x')
        ybottom = pred.get('bottomright').get('y')
        cv2.rectangle(img, (xtop-padding, ytop-padding), (xbottom+padding, ybottom+padding), (0, 0, 0), 3)  
        font_scale = 2
        font = cv2.FONT_HERSHEY_PLAIN
        rectangle_bgr = (0, 0, 0)
        text = "Wally {}%".format(round(float(pred.get('confidence')), 2)*100)
        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = xtop
        text_offset_y = ytop - 30
        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=2)     
    
    return img


def yolo_predict(img, output_path):
    imgcv = cv2.imread(img)
    result = yolo_sensor.return_predict(imgcv)
    labeled_img = highlight_wally(imgcv, result)
    cv2.imwrite('{}_labaled.png'.format(img[:img.rfind('.')]), labeled_img)


def main(image, output_path):
    try:
        yolo_predict(image, output_path)
        print('Wally was found!')
    except:
        print('Sorry, Wally was not found!')


if __name__ == "__main__":
    sys_args = sys.argv

    if len(sys_args) == 2:
        main(sys_args[1], '')
    elif len(sys_args) == 3:
        main(sys_args[1], sys_args[2])
    else: 
        print('Error! You should pass a image. e.g.: python predict wally_118.png')





