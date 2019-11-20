#!/usr/bin/env python3

import numpy as np
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError

WIN_STRIDE = (8,8)
SCALE = None
PADDING = None

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

colors = {
            'white':(255, 255, 255),
            'black':(0,0,0),
            'green':(0, 255, 0),
            'blue':(0,255,255)
}

def angle_from_box(img,box):
    h,w = img.shape[:2]
    (xA, yA, xB, yB) = box
    center_y, center_x = (h-(yA+(yB-yA)/2),(xA+(xB-xA)/2)-w/2)
    if center_x == 0:
        angle = 0
    else: angle = np.round(np.rad2deg(np.arctan2(center_y,center_x))-90.0,decimals=1)

    return angle


def plot_boxes(img, boxes, color='white'):
	global colors
	for box in boxes:
		(xA, yA, xB, yB) = box
		angle = angle_from_box(img,box)
		cv2.rectangle(img, (xA, yA), (xB, yB), colors[color], 5)
		cv2.putText(img, str(angle), (xA, yA-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[color], 2)
	return img


def run_hog_on_img(img):
	boxes, weights = hog.detectMultiScale(img, winStride=WIN_STRIDE)
	boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

	return boxes


class people_publisher():

    def __init__(self):
        rospy.init_node('camera_handler', anonymous=True)
        self.bridge = CvBridge()

        rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, self.zed_callback)
        rospy.Subscriber("/ircam_data", Image, self.ir_callback)

        self.image_pub_ir = rospy.Publisher("/people_hog_ir",Image)
        self.image_pub_zed = rospy.Publisher("/people_hog_zed", Image)

    def ir_callback(self, img_data):
        image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        boxes = run_hog_on_img(image)
        print(boxes)
        img_people = plot_boxes(image, boxes, color='blue')
        self.image_pub_ir.publish(self.bridge.cv2_to_imgmsg(img_people, "bgr8"))

    def zed_callback(self, img_data):
        image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        boxes = run_hog_on_img(image)
        print(boxes)
        img_people = plot_boxes(image, boxes, color='blue')
        self.image_pub_zed.publish(self.bridge.cv2_to_imgmsg(img_people, "bgr8"))


if __name__ == '__main__':
    try:
        detector = people_publisher()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("people_publisher node terminated.")
