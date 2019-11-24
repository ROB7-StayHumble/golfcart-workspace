#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

H_ZED = 720
W_ZED = 1280
H_IR = 600
W_IR = 800

LIDAR_SETZEROTO = 60
RUN_PERSON_DETECTION = True
RUN_LANE_DETECTION = False
RUN_HOUGH = False

if RUN_PERSON_DETECTION:
    winPeople = pg.GraphicsWindow(title="Person detection") # creates a window
    p_lidar_people = winPeople.addPlot(row=1, col=1, title="LIDAR data", labels={'left': 'Range (meters)',
                                                                                 'bottom': 'Angle (degrees)'})  # creates empty space for the plot in the window
    curve_lidar_people = p_lidar_people.plot()  # create an empty "plot" (a curve to plot)
    p_lidar_people.showGrid(x=True, y=True)
    curve_lidar_people.getViewBox().invertX(True)
    curve_lidar_people.getViewBox().setLimits(yMin=0, yMax=80)
    curve_lidar_people.getViewBox().setAspectLocked(True)

    p_ir_people = winPeople.addPlot(row=1, col=2, rowspan=2, title='IR cam')
    imgItem_ir = pg.ImageItem()
    curve_ir = p_ir_people.plot()
    curve_ir.getViewBox().invertY(True)
    curve_ir.getViewBox().setAspectLocked(True)
    p_ir_people.hideAxis('left')
    p_ir_people.hideAxis('bottom')
    p_ir_people.addItem(imgItem_ir)

    p_zed = winPeople.addPlot(row=2, col=1, title='ZED cam')
    imgItem_zed = pg.ImageItem()
    curve_zed = p_zed.plot()
    curve_zed.getViewBox().invertY(True)

    curve_zed.getViewBox().setLimits(xMin=0, xMax=W_ZED)
    curve_zed.getViewBox().setAspectLocked(True)
    p_zed.hideAxis('left')
    p_zed.hideAxis('bottom')
    p_zed.addItem(imgItem_zed)

def angles_of_max_ranges(data_x,data_y):
    peaks, props = find_peaks(data_y, height=30)
    if not len(peaks):
        peaks, props = find_peaks(data_y, height=10)
        peaks = [sorted(peaks,key=lambda x: data_x[x],reverse=True)[0]]
    angles = [data_x[i_max] for i_max in peaks]
    return angles

# Realtime data plot. Each time this function is called, the data display is updated
def update_zed(image):
    image = np.swapaxes(image,0,1)
    imgItem_zed.setImage(image,autoDownsample=True)          # set the curve with this data
    QtGui.QApplication.processEvents()    # you MUST process the plot now

def update_ir(image):
    image_people = np.swapaxes(image,0,1)
    imgItem_ir.setImage(image_people, autoDownsample=True)  # set the curve with this data
    QtGui.QApplication.processEvents()    # you MUST process the plot now

angleLines = []
angles = np.arange(-90,90.5,0.5)
# Realtime data plot. Each time this function is called, the data display is updated
def update_lidar(lidar_ranges):
    angleLines = []
    if RUN_PERSON_DETECTION:
        curve_lidar_people.setData(angles, lidar_ranges)
    QtGui.QApplication.processEvents()    # you MUST process the plot now

class camera_processing():

    def __init__(self):
        rospy.init_node('camera_handler', anonymous=True)
        self.bridge = CvBridge()

        rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, self.zed_callback)
        rospy.Subscriber("/ircam_data", Image, self.ir_callback)
        rospy.Subscriber("/bottom_scan", LaserScan, self.lidar_callback)


    def zed_callback(self, img_data):
        image = self.bridge.imgmsg_to_cv2(img_data, "bgra8")
        update_zed(image)

    def ir_callback(self, img_data):
        image = self.bridge.imgmsg_to_cv2(img_data, "bgra8")
        update_ir(image)

    def lidar_callback(self, lidar_data):
        ranges = np.array(lidar_data.ranges)
        ranges[ranges < lidar_data.range_min] = LIDAR_SETZEROTO
        update_lidar(ranges)


if __name__ == '__main__':
    try:
        detector = camera_processing()
        #rospy.spin()
        pg.QtGui.QApplication.exec_() # you MUST put this at the end

    except rospy.ROSInterruptException:
        rospy.loginfo("Camera_handler node terminated.")
