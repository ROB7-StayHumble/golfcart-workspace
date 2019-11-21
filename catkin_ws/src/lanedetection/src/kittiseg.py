#!/usr/bin/env python3

import numpy as np
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError

from KittiSeg import run_detection

DETECTION_RUNNING = False

def run_kittiseg_on_img(img):
    global DETECTION_RUNNING
    output = run_detection.run_detection(img)
    return output


class kittiseg_publisher():

    def __init__(self):
        rospy.init_node('camera_handler', anonymous=True)
        self.bridge = CvBridge()

        rospy.Subscriber("/ircam_data", Image, self.img_callback)

        self.image_pub = rospy.Publisher("/lane_kittiseg",Image)

    def img_callback(self, img_data):
        global DETECTION_RUNNING
        if not DETECTION_RUNNING:
            DETECTION_RUNNING = True
            image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
            kittiseg_output = run_kittiseg_on_img(image)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(kittiseg_output, "bgr8"))
            DETECTION_RUNNING = False


if __name__ == '__main__':
    try:
        detector = kittiseg_publisher()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("lane_kittiseg node terminated.")
