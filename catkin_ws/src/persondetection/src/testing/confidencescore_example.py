#!/usr/bin/env python3

import glob
import cv2
import numpy as np
from random import Random
from connected_components.connectedcomponents import *
import matplotlib.pyplot as plt

SLOPE = {}
INTERCEPT = {}

img_h = 720
img_w = 1280
GREEN = (0,200,0)
RED = (200,0,0)

folder = "src/persondetection/src/testing/"

def plot_box_with_score(img,h,w,x,y):
    rel_h = h / img_h
    rel_w = w / img_w
    rel_y = y / img_h
    rel_x = x / img_w
    print(rel_h, rel_w, rel_x, rel_y)

    score_yh = calculate_confidence_score(rel_y, rel_h)
    score_aspect_ratio = calculate_confidence_score(rel_h, rel_w)
    score_dimensions = np.round(np.round(score_yh, decimals=2) * np.round(score_aspect_ratio, decimals=2), decimals=2)
    if score_dimensions > 0.5: color = GREEN
    else: color = RED
    print(score_yh, score_aspect_ratio, score_dimensions)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.putText(img, str(score_dimensions), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return img


img = np.zeros((img_h,img_w,3))

h = 100
w = 600
x = 100
y = 150

plot_box_with_score(img,h,w,x,y)

h = 30
w = 10
x = 200
y = 550

plot_box_with_score(img,h,w,x,y)

h = 300
w = 150
x = 500
y = 370

plot_box_with_score(img,h,w,x,y)

h = 150
w = 50
x = 1000
y = 300

plot_box_with_score(img,h,w,x,y)

h = 200
w = 70
x = 900
y = 50

plot_box_with_score(img,h,w,x,y)

plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.savefig(folder + "results/regression/score_boxes.png", bbox_inches='tight')
plt.show()