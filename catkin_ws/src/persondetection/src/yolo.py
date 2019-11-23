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

from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *

from utils.boxes import *

from utils.transformbox import get_boxes_zedframe
from connected_components.connectedcomponents import *

os.chdir('src/persondetection/include/yolov3/')
CFG =       'cfg/yolov3-tiny.cfg'
WEIGHTS =   'weights/yolov3-tiny.weights'
SOURCE =    'data/samples'
OUTPUT =    'output'
DATA =      'data/coco.data'

HALF = False
VIEW_IMG = True
CONF_THRESH = 0.3
NMS_THRESH = 0.5

def detect_from_folder(save_txt=False, save_img=False):
    with torch.no_grad():
        img_size = 416
        out, source, weights, half, view_img = OUTPUT, SOURCE, WEIGHTS, HALF, VIEW_IMG
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(device='')
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Initialize model
        model = Darknet(CFG, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = True
        #     torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=img_size, half=half)
        # else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)
        print(dataset)

        # Get classes and colors
        classes = load_classes(parse_data_cfg(DATA)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # Run inference
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if HALF:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRESH, NMS_THRESH)

            # Apply
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, _, cls in det:
                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (classes[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)

        print('Done. (%.3fs)' % (time.time() - t0))

def detect_from_img(img):
    print("Running detection")
    with torch.no_grad():
        img_size = 416
        out, source, weights, half, view_img = OUTPUT, SOURCE, WEIGHTS, HALF, VIEW_IMG

        # Initialize
        device = torch_utils.select_device(device='')

        # Initialize model
        model = Darknet(CFG, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        save_img = False

        # Get classes and colors
        classes = load_classes(parse_data_cfg(DATA)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        img0s = img  # BGR
        img0 = img
        im0 = img
        assert img0 is not None, 'Image is None'
        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if HALF else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0


        # Run inference
        t0 = time.time()

        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if HALF:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRESH, NMS_THRESH)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        boxes = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, _, cls in det:
                    if view_img and int(cls) == 0:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        box = {'coords':[int(x) for x in [*xyxy]], 'conf':float(conf)}
                        boxes.append(box)

            print('%sDone. (%.3fs)' % ('', time.time() - t))

            # Stream results
            # if view_img:
            #     cv2.imshow('yolo', im0)
            #     cv2.waitKey(0)

    return im0, boxes

from sensor_msgs.msg import Image, LaserScan

FOLDER_EVAL = '/home/nemo/Documents/rob7/combo_eval/'

def get_GT_timestamps():
    IMG_H = 720
    IMG_W = 1280
    gt_timestamps = []
    for filepath in glob.glob(FOLDER_EVAL + "GT_ZED/*.txt"):
        print(filepath)
        timestamp = filepath.split("/")[-1].split("_")[0] # assuming files are named as follows: 1571825140101861494_zed.txt
        print(timestamp)
        with open(filepath) as txtfile:
            gt_image = np.zeros((IMG_H, IMG_W, 3))
            for line in txtfile:
                box = line.strip("\n").split(" ")
                if box[0] == '0':  # person category
                    print(box)
                    x = float(box[1]) * IMG_W
                    y = float(box[2]) * IMG_H
                    h = float(box[4]) * IMG_H
                    w = float(box[3]) * IMG_W

                    pnt_topleft = (int(x - w/2),int(y-h/2))
                    pnt_bottright = (int(x + w/2),int(y+h/2))

                    cv2.rectangle(gt_image, pnt_topleft, pnt_bottright,
                                 color=(255,255,255),
                                 thickness=cv2.FILLED,
                                 lineType=8, shift=0)

            cv2.imwrite(timestamp+'_GT.png',gt_image)
            gt_timestamps.append(timestamp)
    return gt_timestamps

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

        self.boxes_zedframe_class = []
        self.boxes_combined = []

        self.gt_timestamps = get_GT_timestamps()
        print(self.gt_timestamps)
        self.ir_last = None

    def ir_callback(self, img_data):
        print("--> IR")
        image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        self.ir_last = image
        # img_connectedcomp = detect_connected_components(image)
        img_boxes, boxes = detect_from_img(image)
        # print(boxes)
        self.image_pub_ir.publish(self.bridge.cv2_to_imgmsg(img_boxes, "bgr8"))
        # self.image_pub_connectedcomp.publish(self.bridge.cv2_to_imgmsg(img_connectedcomp, "bgr8"))
        boxes_class = [Box(image,xyxy=box['coords'],confidence=box['conf']) for box in boxes]
        confs = [box['conf'] for box in boxes]
        boxes_zedframe = get_boxes_zedframe([box['coords'] for box in boxes])
        for i,box in enumerate(boxes_zedframe):
            # print(box)
            min_x = int(np.min([coord[0] for coord in box]))
            max_x = int(np.max([coord[0] for coord in box]))
            max_y = int(np.max([coord[1] for coord in box]))
            min_y = int(np.min([coord[1] for coord in box]))
            self.boxes_zedframe_class.append({'coords':[min_x,min_y,max_x,max_y],
                                              'conf':confs[i]})
        # if len(self.boxes_zedframe_class)>20:
        #     self.boxes_zedframe_class = self.boxes_zedframe_class[-20:]
        # map = makeConfidenceMapFromBoxes(image,boxes_class)
        # map = cv2.convertScaleAbs(map, alpha=255/map.max())
        # self.image_pub_ir.publish(self.bridge.cv2_to_imgmsg(map, "bgr8"))

    def zed_callback(self, img_data):
        print("--> ZED")
        timestamp = str(img_data.header.stamp)
        print(timestamp)
        image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        img_boxes, boxes = detect_from_img(image)

        self.image_pub_zed.publish(self.bridge.cv2_to_imgmsg(img_boxes, "bgr8"))
        boxes_zed_class = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in boxes]
        if len(boxes_zed_class) > 0:
            self.boxes_combined = boxes_zed_class
            for box in self.boxes_zedframe_class:
                # print(box)
                box_class = Box(image, xyxy=box['coords'], confidence=box['conf'])
                self.boxes_combined.append(box_class)
        else: self.boxes_combined = [Box(image, xyxy=box['coords'], confidence=box['conf']) for box in self.boxes_zedframe_class]
        self.boxes_zedframe_class = []
        map = makeConfidenceMapFromBoxes(image,self.boxes_combined)
        map = cv2.convertScaleAbs(map, alpha=255/map.max())
        self.image_pub_map.publish(self.bridge.cv2_to_imgmsg(map, "bgr8"))
        # if timestamp in self.gt_timestamps:
        print(timestamp)
        print(FOLDER_EVAL + timestamp + '_YOLOboxes.png')
        cv2.imwrite(FOLDER_EVAL + timestamp + '_YOLOboxes.png',img_boxes)
        cv2.imwrite(FOLDER_EVAL + timestamp + '_map.png',map)
        cv2.imwrite(FOLDER_EVAL + timestamp + '_IR.png', self.ir_last)

if __name__ == '__main__':
    try:
        detector = people_yolo_publisher()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("people_yolo_publisher node terminated.")


# detect_from_folder()
