#!/usr/bin/env python3

from utils.boxes import *
from utils.img_utils import *
from utils.transformbox import get_boxes_zedframe
from persondetection.run_yolo import *
from connected_components.connectedcomponents import *

SHOW_PLOTS = True
cam = "IR"

folder = "src/persondetection/src/testing/"

pairs = [("1571746625368332498_ircam.png","1571746625385033721_zed.png","1571746625385033721_GT.png"),
         ("1571746631630210616_ircam.png","1571746631605020001_zed.png","1571746631605020001_GT.png"),
         ("1571746633060068411_ircam.png","1571746633070918853_zed.png","1571746633070918853_GT.png"),
         ("1571746634460540239_ircam.png","1571746634504400900_zed.png","1571746634504400900_GT.png")]

IOU = {}
IOU_avg = {}

modes = ["combo", "ir+zed", "zed", "ir"]

for mode in modes:
    IOU[mode] = []

for pair in pairs:
    ir = cv2.imread(folder+"test_img/ir/"+pair[0])
    ir_1channel = cv2.imread(folder + "test_img/ir/" + pair[0],0)
    zed = cv2.imread(folder + "test_img/zed/" + pair[1])

    gt = cv2.imread(folder + "test_img/gt/" + pair[2])

    ir_cc_img, ir_cc_boxes = detect_connected_components_updated(ir_1channel)
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

    for mode in modes:

        if mode == "combo":
            # total_map = (ir_yolo_map + ir_cc_map + zed_yolo_map)/3
            # ir_yolo_map[np.bitwise_and(ir_cc_map > 0,ir_yolo_map > 0)] = 1
            total_map_max = np.maximum.reduce([ir_yolo_map, ir_cc_map, zed_yolo_map])
            total_map_sum = (ir_yolo_map + ir_cc_map + zed_yolo_map)/3

            zed_yolo_img = cv2.cvtColor(zed_yolo_img, cv2.COLOR_BGR2RGB)
            images = [ir_cc_img,ir_yolo_img,zed_yolo_img,
                      ir_cc_map,ir_yolo_map,zed_yolo_map]
            titles = ["Connected components on IR image","YOLOv3-tiny on IR image", "YOLOv3-tiny on RGB image"]
            fig = plt.figure(num=None, figsize=(12, 4), dpi=300)

            plt.rcParams["axes.titlesize"] = 6
            for i in range(len(images)):
                plt.subplot(2, len(images)/2, i + 1), plt.imshow(images[i], 'gray')
                if i < len(titles):
                    plt.title(titles[i])
                plt.xticks([]), plt.yticks([])

            filename = pair[1].split(".")[-2]
            plt.savefig(folder + "results/total_confmap/" + filename + "_detail.png", bbox_inches='tight')

            images = [total_map_sum,total_map_max]
            titles = ["Total confidence map\n(Average approach)","Total confidence map\n(Element-wise maximum approach)"]
            fig = plt.figure(num=None, figsize=(12, 4), dpi=300)

            plt.rcParams["axes.titlesize"] = 8
            for i in range(len(images)):
                plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])

            plt.savefig(folder + "results/total_confmap/"+filename+"_total.png", bbox_inches='tight')
            # plt.show()

            threshold = 0.7

        elif mode == "ir+zed":
            total_map_max = np.maximum.reduce([ir_yolo_map, zed_yolo_map])
            threshold = 0

        elif mode == "zed":
            total_map_max = zed_yolo_map
            threshold = 0

        elif mode == "ir":
            total_map_max = ir_yolo_map
            threshold = 0

        else: continue

        total_map_bin = total_map_max
        total_map_bin[total_map_max > threshold] = 1
        total_map_bin[total_map_max <= threshold] = 0
        gt_intersection = np.bitwise_and(total_map_bin > 0, gt > 0)
        gt_union = np.bitwise_or(total_map_bin > 0, gt > 0)
        iou = np.sum(gt_intersection)/np.sum(gt_union)
        IOU[mode].append(iou)

        images = [gt, total_map_bin, np.float32(gt_intersection)]
        titles = ["Ground truth",
                  "Binarized confidence map\n(threshold:"+str(threshold)+")",
                  "Overlap between ground truth and confidence map"]
        fig = plt.figure(num=None, figsize=(12, 4), dpi=300)

        plt.rcParams["axes.titlesize"] = 8
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.savefig(folder + "results/gt_eval/" + mode + "/" + filename + ".png", bbox_inches='tight')
        #plt.show()

print(IOU)

for key, val in IOU.items():
    IOU_avg[key] = np.average(val)

print(IOU_avg)

