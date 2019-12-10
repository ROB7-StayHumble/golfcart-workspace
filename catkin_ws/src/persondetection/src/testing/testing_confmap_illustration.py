#!/usr/bin/env python3

from utils.boxes import *
from utils.img_utils import *
from utils.transformbox import get_boxes_zedframe
from persondetection.run_yolo import *
from connected_components.connectedcomponents import *

SHOW_PLOTS = False
SAVE_PLOTS = False
cam = "IR"

folder = "src/persondetection/src/testing/"

ir_images, zed_images, gt_images = [], [], []
with open(folder+"test_img/ir.txt") as f:
    for line in f:
        ir_images.append(line.strip('\n'))
with open(folder+"test_img/zed.txt") as f:
    for line in f:
        zed_images.append(line.strip('\n'))
with open(folder+"test_img/gt.txt") as f:
    for line in f:
        gt_images.append(line.strip('\n'))

pairs = list(zip(ir_images, zed_images, gt_images))

thresholds = np.linspace(0,1,num=50,endpoint=False)

for i,pair in enumerate(pairs):
    if i < 500:
        print(pair)
        ir = cv2.imread(folder+"test_img/ir/"+pair[0])
        ir_1channel = cv2.imread(folder + "test_img/ir/" + pair[0],0)
        zed = cv2.imread(folder + "test_img/zed/" + pair[1])

        gt = cv2.imread(folder + "test_img/gt/" + pair[2])

        ir_cc_thresh_img, ir_cc_grad_img, ir_cc_boxes = detect_connected_components_updated(ir_1channel)
        ir_yolo_img, ir_yolo_boxes = detect_from_img(ir)
        zed_yolo_img, zed_yolo_boxes = detect_from_img(zed)

        ir_cc_boxes_zedframe = get_boxes_zedframe(
            [Box(blank_ir_3D, xyxy=box, confidence=1) for box in ir_cc_boxes])
        zed_yolo_boxes_zedframe = [Box(blank_zed_3D, xyxy=box['coords'], confidence=1) for box in zed_yolo_boxes]
        ir_yolo_boxes_zedframe = get_boxes_zedframe(
            [Box(blank_ir_3D, xyxy=box['coords'], confidence=1) for box in ir_yolo_boxes])

        ir_yolo_map = makeConfidenceMapFromBoxes(blank_zed_3D, ir_yolo_boxes_zedframe)
        ir_cc_map = makeConfidenceMapFromBoxes(blank_zed_3D, ir_cc_boxes_zedframe)
        zed_yolo_map = makeConfidenceMapFromBoxes(blank_zed_3D, zed_yolo_boxes_zedframe)

        total_map_max = np.maximum.reduce([ir_yolo_map, ir_cc_map, zed_yolo_map])
        total_map_sum = (ir_yolo_map + ir_cc_map + zed_yolo_map)/3

        zed_yolo_img = cv2.cvtColor(zed_yolo_img, cv2.COLOR_BGR2RGB)

        filename = pair[1].split(".")[-2]
        # plt.savefig(folder + "results/total_confmap/" + filename + "_detail.png", bbox_inches='tight')

        # ZED image
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(zed)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_zed.png", bbox_inches='tight')

        #IR image
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(ir)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_ir.png", bbox_inches='tight')

        #IR CC - thresh
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(ir_cc_thresh_img)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_ir_cc_thresh.png", bbox_inches='tight')

        #IR CC - grad
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(ir_cc_grad_img)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_ir_cc_grad.png", bbox_inches='tight')

        #IR map - yolo
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(ir_yolo_map)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_ir_yolo_map.png", bbox_inches='tight')

        #IR map - CC
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(ir_cc_map)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_ir_cc_map.png", bbox_inches='tight')

        #ZED map - YOLO
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(zed_yolo_map)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_zed_yolo_map.png", bbox_inches='tight')

        #Total map - max
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(total_map_max)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/"+filename+"_total_map_max.png", bbox_inches='tight')

        # Total map - sum
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
        plt.imshow(total_map_sum)
        plt.xticks([]), plt.yticks([])
        plt.savefig(folder + "results/illustration/" + filename + "_total_map_sum.png", bbox_inches='tight')

        # plt.show()


        plt.close('all')
