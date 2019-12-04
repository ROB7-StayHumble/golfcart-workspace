#!/usr/bin/env python3

import glob
import cv2
import numpy as np
from random import Random
from connected_components.connectedcomponents import *

SLOPE = {}
INTERCEPT = {}

# ZED
# SLOPE['y_height'] = 1.838772993923768
# INTERCEPT['y_height'] = -0.592406200468761
# SLOPE['aspect_ratio'] = 0.2193557959956911
# INTERCEPT['aspect_ratio'] = 0.003921820805625677

# def distance_from_line(x,y,a,b):
#     predicted_y = x*a + b
#
#     diff = np.abs(y - predicted_y)
#     return diff
#
# def calculate_confidence_score(box_x,box_y,model_slope,model_intercept):
#
#     diff_max = np.max([ distance_from_line(0,0,model_slope,model_intercept),
#                         distance_from_line(0,1,model_slope,model_intercept),
#                         distance_from_line(1,1,model_slope,model_intercept),
#                         distance_from_line(1,0,model_slope,model_intercept)])
#     diff = distance_from_line(box_x,box_y,model_slope,model_intercept)
#     print(diff,diff_max)
#     score = 1 - diff/(diff_max)
#     #print(diff)
#     #print(box_x,box_y,model_slope,model_intercept,score)
#     return np.round(score,decimals=2)


img_h = 720
img_w = 1280
GREEN = (0,200,0)
RED = (0,0,200)

img = np.zeros((img_h,img_w,3))

h = 100
w = 600
x = 100
y = 150

rel_h = h / img_h
rel_w = w / img_w
rel_y = y / img_h
rel_x = x /img_w
print(rel_h,rel_w,rel_x,rel_y)

score_yh = calculate_confidence_score(rel_y, rel_h)
score_aspect_ratio = calculate_confidence_score(rel_h, rel_w)
score_dimensions = np.round(np.round(score_yh, decimals=2) * np.round(score_aspect_ratio, decimals=2),decimals=2)
print(score_yh, score_aspect_ratio, score_dimensions)
cv2.rectangle(img,(x,y),(x+w,y+h),RED,3)
cv2.putText(img, str(score_dimensions), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2)

h = 30
w = 10
x = 200
y = 550

rel_h = h / img_h
rel_w = w / img_w
rel_y = y / img_h
rel_x = x /img_w
print(rel_h,rel_w,rel_x,rel_y)

score_yh = calculate_confidence_score(rel_y, rel_h)
score_aspect_ratio = calculate_confidence_score(rel_h, rel_w)
score_dimensions = np.round(np.round(score_yh, decimals=2) * np.round(score_aspect_ratio, decimals=2),decimals=2)
print(score_yh, score_aspect_ratio, score_dimensions)
cv2.rectangle(img,(x,y),(x+w,y+h),RED,3)
cv2.putText(img, str(score_dimensions), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, RED, 2)

h = 300
w = 100
x = 500
y = 370

rel_h = h / img_h
rel_w = w / img_w
rel_y = y / img_h
rel_x = x /img_w
print(rel_h,rel_w,rel_x,rel_y)

score_yh = calculate_confidence_score(rel_y, rel_h)
score_aspect_ratio = calculate_confidence_score(rel_h, rel_w)
score_dimensions = np.round(np.round(score_yh, decimals=2) * np.round(score_aspect_ratio, decimals=2),decimals=2)
print(score_yh, score_aspect_ratio, score_dimensions)
cv2.rectangle(img,(x,y),(x+w,y+h),GREEN,3)
cv2.putText(img, str(score_dimensions), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, GREEN, 2)

h = 150
w = 50
x = 1000
y = 300

rel_h = h / img_h
rel_w = w / img_w
rel_y = y / img_h
rel_x = x /img_w
print(rel_h,rel_w,rel_x,rel_y)

score_yh = calculate_confidence_score(rel_y, rel_h)
score_aspect_ratio = calculate_confidence_score(rel_h, rel_w)
score_dimensions = np.round(np.round(score_yh, decimals=2) * np.round(score_aspect_ratio, decimals=2),decimals=2)
print(score_yh, score_aspect_ratio, score_dimensions)
cv2.rectangle(img,(x,y),(x+w,y+h),GREEN,3)
cv2.putText(img, str(score_dimensions), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, GREEN, 2)

cv2.imshow('la',img)
cv2.waitKey(0)