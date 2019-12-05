#!/usr/bin/env python3

from utils.boxes import *
from utils.img_utils import *
from utils.transformbox import get_boxes_zedframe
from persondetection.run_yolo import *

SHOW_PLOTS = True
cam = "IR"

folder = "src/persondetection/src/testing/"

print(os.getcwd())
if cam == "IR":
    for imgpath in glob.glob(folder+"test_img/ircam*.png"):
        #thresholding(imgpath)
        print(imgpath)
        img = cv2.imread(imgpath)

        cimg, boxes = detect_from_img(img)
        boxes_zedframe = get_boxes_zedframe([Box(blank_ir_3D,xyxy=box['coords'],confidence=1) for box in boxes])
        boxes_img = plot_boxes_on_img(blank_zed_3D,[box.xyxy for box in boxes_zedframe])
        map = makeConfidenceMapFromBoxes(blank_zed_3D, boxes_zedframe)

        images = [cimg, boxes_img, map]
        titles = ["YOLOv3-tiny detection","Perspective transformation to\nZED camera frame", "Confidence map"]
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)

        plt.rcParams["axes.titlesize"] = 4
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        fig.tight_layout()
        # plt.imshow(map)

        if SHOW_PLOTS: plt.show()
        # histo(imgpath)
elif cam == "ZED":
    for imgpath in glob.glob(folder + "test_img/zedcam*.png"):
        # thresholding(imgpath)
        print(imgpath)
        img = cv2.imread(imgpath)

        cimg, boxes = detect_from_img(img)
        boxes_class = [Box(blank_zed_3D, xyxy=box['coords'], confidence=1) for box in boxes]
        # boxes_img = plot_boxes_on_img(blank_ir_3D, [box.xyxy for box in boxes_class])
        map = makeConfidenceMapFromBoxes(blank_zed_3D, boxes_class)

        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
        images = [cimg, map]
        titles = ["YOLOv3-tiny detection", "Confidence map"]
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)

        plt.rcParams["axes.titlesize"] = 4
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        fig.tight_layout()
        # plt.imshow(map)

        if SHOW_PLOTS: plt.show()
        # histo(imgpath)
