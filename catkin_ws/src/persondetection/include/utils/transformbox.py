#!/usr/bin/env python3

import numpy as np
import cv2
import scipy.io

from utils.img_utils import *
import os
from utils.boxes import Box

path = '/home/zacefron/Desktop/golfcart-workspace/catkin_ws/src/persondetection/include/utils/'
folder = '1571825077852250111'
# load transformation info
points_dict = scipy.io.loadmat(path+'tf_points/'+folder+'/points.mat')
tf = points_dict['tform']
boxes = points_dict['boxes']

def get_box_zedframe(box,tform=tf):
	xA, yA, xB, yB = box.xyxy
	w = xB - xA
	h = yB - yA
	box_corners = np.float32([[[xA,yA],[xB,yA],[xB,yB],[xA,yB]]])
	#print(box_corners)
	box_warped = cv2.perspectiveTransform(box_corners, tform)
	min_x = int(np.min([coord[0] for coord in box_warped[0]]))
	max_x = int(np.max([coord[0] for coord in box_warped[0]]))
	max_y = int(np.max([coord[1] for coord in box_warped[0]]))
	min_y = int(np.min([coord[1] for coord in box_warped[0]]))
	newbox = Box(img=blank_zed_3D, xyxy=[min_x,min_y,max_x,max_y], confidence=1)
	return newbox

def get_boxes_zedframe(boxes,tform=tf):
	n = 0
	boxes_tformed = []
	for box in boxes:
		newbox = get_box_zedframe(box)
		# box.transform(xyxy=[min_x,min_y,max_x,max_y])
		boxes_tformed.append(newbox)
		n += 1
	return boxes_tformed

def show_transform_box(tform,boxes,ir,zed):

	# create resizable windows for plotting
	cv2.namedWindow('IR',cv2.WINDOW_NORMAL)
	cv2.namedWindow('ZED',cv2.WINDOW_NORMAL)

	for box in boxes:
		box_corners = np.array([box])

		# plot bounding box for IR frame
		ir = plot_polygons(ir,box_corners[0])
		cv2.imshow("IR",ir)

		# transform bounding box coordinates to ZED frame
		box_warped = cv2.perspectiveTransform(box_corners, tform)

		# plot bounding box for ZED frame
		zed = plot_polygons(zed,box_warped[0])

	cv2.imshow("ZED",zed)

	key = cv2.waitKey(0) & 0xFF
	if key == ord("q"): cv2.destroyAllWindows()

#
#
# ir = cv2.imread('ircam'+folder+'.png')
# zed = cv2.imread('zedcam'+folder+'.png')
#show_transform_box(tf,boxes,ir,zed)
