#!/usr/bin/env python2
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
RUN_LANE_DETECTION = True
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

if RUN_LANE_DETECTION:
    winLane = pg.GraphicsWindow(title="Lane detection") # creates a window
    p_lidar_lane = winLane.addPlot(row=1, col=1, rowspan=1, title="LIDAR data",labels={'left':'Range (meters)','bottom':'Angle (degrees)'})  # creates empty space for the plot in the window
    curve_lidar_lane = p_lidar_lane.plot()
    curve_lidar_smooth = p_lidar_lane.plot()                        # create an empty "plot" (a curve to plot)
    p_lidar_lane.showGrid(x=True,y=True)
    curve_lidar_lane.getViewBox().invertX(True)
    curve_lidar_lane.getViewBox().setLimits(yMin=0,yMax=80)
    curve_lidar_lane.getViewBox().setAspectLocked(True)
    #p_lidar.setYRange(0, 50, padding=0)
    curve_lidar_smooth.setPen(pg.mkPen({'color': (100, 255, 255, 150), 'width': 4}))

    p_ir_lane = winLane.addPlot(row=1, col=2, rowspan=2, title = 'IR cam')
    p_ir_edges = winLane.addPlot(row=2, col=1, rowspan=1, title='IR cam')
    imgItem_ir_lane = pg.ImageItem()
    curve_ir_lane = p_ir_lane.plot()
    curve_ir_lane.getViewBox().invertY(True)
    curve_ir_lane.getViewBox().setAspectLocked(True)
    p_ir_lane.addItem(imgItem_ir_lane)

    imgItem_ir_edges = pg.ImageItem()
    curve_ir_edges = p_ir_edges.plot()
    curve_ir_edges.getViewBox().invertY(True)
    curve_ir_edges.getViewBox().setAspectLocked(True)
    p_ir_edges.addItem(imgItem_ir_edges)

def show_lane_direction(angles):
    for line in angleLines:
        p_ir_lane.removeItem(line)
    for angle in angles:
        angleLine = pg.InfiniteLine(pos=(W_IR/2,H_IR), angle=90-angle, movable=False, pen=pg.mkPen({'color': (0, 255, 0, 100), 'width': 4}))
        angleLines.append(angleLine)
        p_ir_lane.addItem(angleLine, ignoreBounds=True)

def angles_of_max_ranges(data_x,data_y):
    peaks, props = find_peaks(data_y, height=30)
    if not len(peaks):
        peaks, props = find_peaks(data_y, height=10)
        peaks = [sorted(peaks,key=lambda x: data_x[x],reverse=True)[0]]
    angles = [data_x[i_max] for i_max in peaks]
    return angles

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth_lidar_data(method,data_x,data_y):
    if method == 'poly':
        poly = np.polyfit(data_x,data_y,15)
        poly_y = np.poly1d(poly)(data_x)
        return poly_y
    elif method == 'butter':
        cutoff = 1500
        fs = 100000
        return butter_lowpass_filtfilt(data_y, cutoff, fs)

# Realtime data plot. Each time this function is called, the data display is updated
def update_zed(image):
    if RUN_PERSON_DETECTION:
        image = np.swapaxes(image,0,1)
        imgItem_zed.setImage(image,autoDownsample=True)          # set the curve with this data
    QtGui.QApplication.processEvents()    # you MUST process the plot now

def update_ir(image):
    if RUN_PERSON_DETECTION:
        image_people = np.swapaxes(image,0,1)
        imgItem_ir.setImage(image_people, autoDownsample=True)  # set the curve with this data
    if RUN_LANE_DETECTION:
        image_ir_inv = np.swapaxes(image, 0, 1)
        imgItem_ir_lane.setImage(image_ir_inv, autoDownsample=True)  # set the curve with this data
        if RUN_HOUGH:
            detected = draw_lanes(image)
            image_edges_inv = np.swapaxes(detected, 0, 1)
            imgItem_ir_edges.setImage(image_edges_inv, autoDownsample=True)  # set the curve with this data
    QtGui.QApplication.processEvents()    # you MUST process the plot now

angleLines = []
angles = np.arange(-90,90.5,0.5)
# Realtime data plot. Each time this function is called, the data display is updated
def update_lidar(lidar_ranges):
    angleLines = []
    if RUN_PERSON_DETECTION:
        curve_lidar_people.setData(angles, lidar_ranges)
    if RUN_LANE_DETECTION:
        smooth = smooth_lidar_data('butter',angles,lidar_ranges)
        curve_lidar_lane.setData(angles, lidar_ranges)
        curve_lidar_smooth.setData(angles,smooth)
        lane_angles = angles_of_max_ranges(angles,smooth)
        #show_lane_direction(lane_angles)
    QtGui.QApplication.processEvents()    # you MUST process the plot now

class camera_processing():

    def __init__(self):
        rospy.init_node('camera_handler', anonymous=True)
        self.bridge = CvBridge()

        rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, self.zed_callback)
        rospy.Subscriber("/ircam_data", Image, self.ir_callback)
        rospy.Subscriber("/bottom_scan", LaserScan, self.lidar_callback)


    def zed_callback(self, img_data):
        try:
            image = self.bridge.imgmsg_to_cv2(img_data, "bgra8")
            update_zed(image)
        except Exception as e:
            print(e)

    def ir_callback(self, img_data):
        try:
            image = self.bridge.imgmsg_to_cv2(img_data, "bgra8")
            update_ir(image)
        except Exception as e:
            print(e)

    def lidar_callback(self, lidar_data):
        try:
            ranges = np.array(lidar_data.ranges)
            ranges[ranges == 0] = LIDAR_SETZEROTO
            update_lidar(ranges)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    try:
        detector = camera_processing()
        #rospy.spin()
        pg.QtGui.QApplication.exec_() # you MUST put this at the end

    except rospy.ROSInterruptException:
        rospy.loginfo("Camera_handler node terminated.")
