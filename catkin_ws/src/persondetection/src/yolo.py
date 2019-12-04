#!/usr/bin/env python3

import numpy as np
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
import os
import argparse
from sys import platform
import glob

from persondetection.run_yolo import *

from utils.boxes import *

from utils.transformbox import get_boxes_zedframe
from connected_components.connectedcomponents import *

from sensor_msgs.msg import Image, LaserScan

# FOLDER_EVAL = '/home/zacefron/Desktop/YOLO annotations(400)-20191123T105218Z-001/maps_connectedcomp_1125/'

def get_GT_timestamps():
    IMG_H = 720
    IMG_W = 1280
    gt_timestamps = []
    with open(FOLDER_EVAL + "GT_pairs/pairs.txt","r") as pairstxt:
        for line in pairstxt:
            timestamp = line.split(' ')[-1].strip('\n')
            gt_timestamps.append(timestamp)
    return gt_timestamps

MODES = {   'ZED_ONLY':'/home/zacefron/Desktop/YOLO annotations(400)-20191123T105218Z-001/maps_zedonly_1125/',
            'ZED_IR':'/home/zacefron/Desktop/YOLO annotations(400)-20191123T105218Z-001/maps_zedir_1125/',
            'ZED_CONNECTEDCOMP':'/home/zacefron/Desktop/YOLO annotations(400)-20191123T105218Z-001/maps_connectedcomp_1125/'
        }
MODE = 'ZED_CONNECTEDCOMP'
FOLDER_EVAL = MODES[MODE]

class people_yolo_publisher():

    def __init__(self):
        rospy.init_node('camera_handler', anonymous=True)
        self.bridge = CvBridge()

        rospy.Subscriber("/ircam_data", Image, self.ir_callback)
        rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, self.zed_callback)

        self.image_pub_ir = rospy.Publisher("/people_yolo_ir",Image)
        self.image_pub_zed = rospy.Publisher("/people_yolo_zed", Image)
        self.image_pub_zedir = rospy.Publisher("/people_yolo_zedir", Image)
        self.image_pub_map = rospy.Publisher("/people_yolo_map", Image)
        self.image_pub_connectedcomp = rospy.Publisher("/people_yolo_connectedcomp", Image)

        self.boxes_zedframe_ir_class = []
        self.boxes_zedframe_cc_class = []
        self.boxes_combined = []

        self.gt_timestamps = get_GT_timestamps()
        print(len(self.gt_timestamps),self.gt_timestamps)
        self.ir_last = None
        self.connectedcomp_last = None
        self.last_callback = None

    def ir_callback(self, img_data):
        if MODE == 'ZED_ONLY':
            return
        if self.last_callback == 'zed':
            self.boxes_zedframe_cc_class = []
            self.boxes_zedframe_ir_class = []
            print("emptying IR boxes")
        print("--> IR")
        image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")


        img_boxes, boxes_yolo = detect_from_img(image.copy())
        if MODE == 'ZED_CONNECTEDCOMP':
            img_connectedcomp, boxes_connectedcomp = detect_connected_components(image.copy())
            # boxes = np.concatenate((boxes_yolo,boxes_connectedcomp))
            self.connectedcomp_last = img_connectedcomp
        self.ir_last = img_boxes
        self.connectedcomp_last = img_connectedcomp
        self.image_pub_ir.publish(self.bridge.cv2_to_imgmsg(img_boxes, "bgr8"))
        if MODE == 'ZED_CONNECTEDCOMP':
            self.image_pub_connectedcomp.publish(self.bridge.cv2_to_imgmsg(img_connectedcomp, "bgr8"))
            confs_cc = [box['conf'] for box in boxes_connectedcomp]
            boxes_zedframe_cc = get_boxes_zedframe([box['coords'] for box in boxes_connectedcomp])
            self.boxes_zedframe_ir_class = []
            for i, box in enumerate(boxes_zedframe_cc):
                # print(box)
                min_x = int(np.min([coord[0] for coord in box]))
                max_x = int(np.max([coord[0] for coord in box]))
                max_y = int(np.max([coord[1] for coord in box]))
                min_y = int(np.min([coord[1] for coord in box]))
                self.boxes_zedframe_cc_class.append({'coords': [min_x, min_y, max_x, max_y],
                                                     'conf': confs_cc[i]})

        # boxes_class = [Box(image,xyxy=box['coords'],confidence=box['conf']) for box in boxes]
        confs_ir = [box['conf'] for box in boxes_yolo]
        boxes_zedframe_ir = get_boxes_zedframe([box['coords'] for box in boxes_yolo])
        self.boxes_zedframe_cc_class = []
        self.boxes_zedframe_ir_class = []
        for i,box in enumerate(boxes_zedframe_ir):
            # print(box)
            min_x = int(np.min([coord[0] for coord in box]))
            max_x = int(np.max([coord[0] for coord in box]))
            max_y = int(np.max([coord[1] for coord in box]))
            min_y = int(np.min([coord[1] for coord in box]))
            self.boxes_zedframe_ir_class.append({'coords':[min_x,min_y,max_x,max_y],
                                              'conf':confs_ir[i]})

        # if len(self.boxes_zedframe_class)>20:
        #     self.boxes_zedframe_class = self.boxes_zedframe_class[-20:]
        # map = makeConfidenceMapFromBoxes(image,boxes_class)
        # map = cv2.convertScaleAbs(map, alpha=255/map.max())
        # self.image_pub_ir.publish(self.bridge.cv2_to_imgmsg(map, "bgr8"))
        self.last_callback = 'ir'

    def zed_callback(self, img_data):
        timestamp = str(img_data.header.stamp)
        image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        img_boxes, boxes = detect_from_img(image)

        self.image_pub_zed.publish(self.bridge.cv2_to_imgmsg(img_boxes, "bgr8"))
        boxes_zed_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in boxes]
        map_zed = makeConfidenceMapFromBoxes(image, boxes_zed_class)

        # if not (MODE == 'ZED_ONLY'):

            # if len(boxes_zed_class) > 0:
            #     # self.boxes_combined = boxes_zed_class
            #     self.boxes_combined = np.concatenate((boxes_zed_class,boxes_ir_class))
            # else: self.boxes_combined = boxes_ir_class


        if MODE == 'ZED_CONNECTEDCOMP':
            boxes_ir_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in
                              self.boxes_zedframe_ir_class]
            boxes_cc_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in
                              self.boxes_zedframe_cc_class]
            map_ir = makeConfidenceMapFromBoxes(image,boxes_ir_class)
            map_cc = makeConfidenceMapFromBoxes(image, boxes_cc_class)
            map = map_ir + map_zed + map_cc
            map = cv2.convertScaleAbs(map, alpha=255 / 3)
        elif MODE == 'ZED_IR':
            boxes_ir_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in
                              self.boxes_zedframe_ir_class]
            map_ir = makeConfidenceMapFromBoxes(image, boxes_ir_class)
            map = map_ir + map_zed
            map = cv2.convertScaleAbs(map, alpha=255 / 2)
        else:
            map = map_zed
            map = cv2.convertScaleAbs(map, alpha=255)
        mapmax = np.max([1,map.max()])

        self.image_pub_map.publish(self.bridge.cv2_to_imgmsg(map, "bgr8"))
        # if timestamp in self.gt_timestamps:
        print(timestamp)
        print(FOLDER_EVAL + timestamp + '_YOLOboxes.png')
        cv2.imwrite(FOLDER_EVAL + timestamp + '_YOLOboxes.png',img_boxes)
        cv2.imwrite(FOLDER_EVAL + timestamp + '_map.png',map)
        cv2.imwrite(FOLDER_EVAL + timestamp + '_IR.png', self.ir_last)
        cv2.imwrite(FOLDER_EVAL + timestamp + '_connectedcomp.png', self.connectedcomp_last)
        # self.ir_last = None
        self.last_callback = 'zed'

if __name__ == '__main__':
    try:
        detector = people_yolo_publisher()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("people_yolo_publisher node terminated.")


# detect_from_folder()
