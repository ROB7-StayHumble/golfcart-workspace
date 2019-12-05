#!/usr/bin/env python3

from utils.boxes import *
from utils.img_utils import *
from utils.transformbox import get_boxes_zedframe
from persondetection.run_yolo import *
from connected_components.connectedcomponents import *

SHOW_PLOTS = True
cam = "IR"

folder = "src/persondetection/src/testing/"

pairs = [("1571746625368332498_ircam.png","1571746625385033721_zed.png")]

for pair in pairs:
    ir = cv2.imread(folder+"test_img/ir/"+pair[0])
    ir_1channel = cv2.imread(folder + "test_img/ir/" + pair[0],0)
    zed = cv2.imread(folder + "test_img/zed/" + pair[1])

    ir_cc_img, ir_cc_boxes = detect_connected_components_updated(ir_1channel)
    ir_yolo_img, ir_yolo_boxes = detect_from_img(ir)
    zed_yolo_img, zed_yolo_boxes = detect_from_img(zed)

    ir_yolo_boxes_zedframe = get_boxes_zedframe([Box(blank_ir_3D, xyxy=box['coords'], confidence=1) for box in ir_yolo_boxes])
    ir_cc_boxes_zedframe = get_boxes_zedframe([Box(blank_ir_3D, xyxy=box, confidence=1) for box in ir_cc_boxes])
    zed_yolo_boxes_zedframe = [Box(blank_zed_3D, xyxy=box['coords'], confidence=1) for box in zed_yolo_boxes]

    ir_yolo_map = makeConfidenceMapFromBoxes(blank_zed_3D, ir_yolo_boxes_zedframe)
    ir_cc_map = makeConfidenceMapFromBoxes(blank_zed_3D, ir_cc_boxes_zedframe)
    zed_yolo_map = makeConfidenceMapFromBoxes(blank_zed_3D, zed_yolo_boxes_zedframe)

    total_map = (ir_yolo_map + ir_cc_map + zed_yolo_map)/3

    images = [ir_cc_img,ir_yolo_img,zed_yolo_img,
              ir_cc_map,ir_yolo_map,zed_yolo_map]
    titles = ["YOLOv3-tiny detection", "Confidence map"]
    fig = plt.figure(num=None, figsize=(12, 4), dpi=300)

    plt.rcParams["axes.titlesize"] = 4
    for i in range(len(images)):
        plt.subplot(2, len(images)/2, i + 1), plt.imshow(images[i], 'gray')
        # plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
    plt.imshow(total_map, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()