#!/usr/bin/env python2

import rosbag
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import glob
from utils.transformbox import get_boxes_zedframe
from connected_components.connectedcomponents import *
import os
from utils.boxes import *

from persondetection.run_yolo import *

os.chdir('src/persondetection/include/yolov3/')
CFG =       'cfg/yolov3-tiny.cfg'
WEIGHTS =   'weights/yolov3-tiny.weights'
SOURCE =    'data/samples'
OUTPUT =    'output'
DATA =      'data/coco.data'

bridge = CvBridge()

bag = rosbag.Bag('/home/zacefron/BAGS/testing2.bag')
FOLDER_EVAL = '/home/zacefron/Desktop/YOLO annotations(400)-20191123T105218Z-001/maps_connectedcomp_1125/'

def get_GT_timestamps():
    gt_timestamps = []
    for filepath in glob.glob(FOLDER_EVAL + "GT_ZED/*.txt"):
        print(filepath)
        timestamp = filepath.split("/")[-1].split("_")[0] # assuming files are named as follows: 1571825140101861494_zed.txt
        print(timestamp)
        gt_timestamps.append(timestamp)
    return gt_timestamps

boxes_zedframe_class = []
boxes_combined = []

# self.gt_timestamps = get_GT_timestamps()
# print(self.gt_timestamps)
ir_last = None
connectedcomp_last = None
last_callback = None

timestamps = get_GT_timestamps()
n_ir=0
n_zed=0
for topic, msg, t in bag.read_messages():
    if topic=='ircam_data' or topic == '/ircam_data':
        boxes_zedframe_class = []
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        print(t,topic,image.shape)
        img_connectedcomp, boxes_connectedcomp = detect_connected_components(image.copy())
        img_boxes, boxes_yolo = detect_from_img(image.copy())
        boxes = np.concatenate((boxes_yolo,boxes_connectedcomp))
        ir_last = img_boxes
        connectedcomp_last = img_connectedcomp

        # boxes_class = [Box(image,xyxy=box['coords'],confidence=box['conf']) for box in boxes]
        confs = [box['conf'] for box in boxes]
        boxes_zedframe = get_boxes_zedframe([box['coords'] for box in boxes])
        for i,box in enumerate(boxes_zedframe):
            # print(box)
            min_x = int(np.min([coord[0] for coord in box]))
            max_x = int(np.max([coord[0] for coord in box]))
            max_y = int(np.max([coord[1] for coord in box]))
            min_y = int(np.min([coord[1] for coord in box]))
            boxes_zedframe_class.append({'coords':[min_x,min_y,max_x,max_y],
                                              'conf':confs[i]})

    elif topic=='/zed_node/rgb/image_rect_color' and t in timestamps:
        img_data = bridge.imgmsg_to_cv2(msg, "bgr8")
        print(t,topic,image.shape)
        print("--> ZED")
        image = bridge.imgmsg_to_cv2(img_data, "bgr8")
        img_boxes, boxes = detect_from_img(image)

        boxes_zed_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in boxes]
        boxes_ir_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in boxes_zedframe_class]
        if len(boxes_zed_class) > 0:
            # self.boxes_combined = boxes_zed_class
            boxes_combined = np.concatenate((boxes_zed_class, boxes_ir_class))
        else:
            boxes_combined = boxes_ir_class

        map_zed = makeConfidenceMapFromBoxes(image, boxes_zed_class)
        map_ir = makeConfidenceMapFromBoxes(image, boxes_ir_class)
        map = map_ir + map_zed
        map = cv2.convertScaleAbs(map, alpha=255 / map.max())

        # if timestamp in self.gt_timestamps:

        print(FOLDER_EVAL + t + '_YOLOboxes.png')
        cv2.imwrite(FOLDER_EVAL + t + '_YOLOboxes.png', img_boxes)
        cv2.imwrite(FOLDER_EVAL + t + '_map.png', map)
        cv2.imwrite(FOLDER_EVAL + t + '_IR.png', ir_last)
        cv2.imwrite(FOLDER_EVAL + t + '_connectedcomp.png', connectedcomp_last)

bag.close()
cv2.destroyAllWindows()