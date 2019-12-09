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

pairs = [("1571746625368332498_ircam.png","1571746625385033721_zed.png","1571746625385033721_GT.png"),
         ("1571746631630210616_ircam.png","1571746631605020001_zed.png","1571746631605020001_GT.png"),
         ("1571746633060068411_ircam.png","1571746633070918853_zed.png","1571746633070918853_GT.png"),
         ("1571746634460540239_ircam.png","1571746634504400900_zed.png","1571746634504400900_GT.png"),
         ("1571746637887406440_ircam.png","1571746637903143927_zed.png","1571746637903143927_GT.png"),
         ("1571746638219263494_ircam.png","1571746638201841864_zed.png","1571746638201841864_GT.png"),
         ("1571746650581633948_ircam.png","1571746650603904756_zed.png","1571746650603904756_GT.png"),
         ("1571746651846831705_ircam.png","1571746651669000660_zed.png","1571746651669000660_GT.png"),
         ("1571746652679496222_ircam.png","1571746652703256760_zed.png","1571746652703256760_GT.png"),
         ("1571746652813228854_ircam.png","1571746652834700756_zed.png","1571746652834700756_GT.png"),
         ("1571746653079627742_ircam.png","1571746653035193106_zed.png","1571746653035193106_GT.png")]

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

IOUs = {}
IOU_avg = {}

precisions = {}
precision_avg = {}

recalls = {}
recall_avg = {}

modes = ["combo", "ir+zed", "zed", "ir"]

for mode in modes:
    IOUs[mode] = []
    precisions[mode] = []
    recalls[mode] = []

for i,pair in enumerate(pairs):
    if i < 500:
        print(pair)
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
                total_map = total_map_max

                zed_yolo_img = cv2.cvtColor(zed_yolo_img, cv2.COLOR_BGR2RGB)

                if SAVE_PLOTS or SHOW_PLOTS:
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

                threshold = 0.8

            elif mode == "ir+zed":
                total_map_max = np.maximum.reduce([ir_yolo_map, zed_yolo_map])
                total_map_sum = (ir_yolo_map + zed_yolo_map)/2
                total_map = total_map_max
                threshold = 0

            elif mode == "zed":
                total_map = zed_yolo_map
                threshold = 0

            elif mode == "ir":
                total_map = ir_yolo_map
                threshold = 0

            else: continue

            total_map_bin = total_map
            total_map_bin[total_map > threshold] = 1
            total_map_bin[total_map <= threshold] = 0
            gt_intersection = np.bitwise_and(total_map_bin > 0, gt > 0)
            gt_union = np.bitwise_or(total_map_bin > 0, gt > 0)
            iou = np.sum(gt_intersection)/np.sum(gt_union)
            IOUs[mode].append(iou)

            TP = np.sum(gt_intersection)
            FP = np.sum(np.bitwise_and(total_map_bin > 0, gt == 0))
            FN = np.sum(np.bitwise_and(total_map_bin == 0, gt > 0))
            if TP+FP > 0:
                precision = TP/(TP+FP)
            else:
                precision = 0
            recall = TP/(TP+FN)

            precisions[mode].append(precision)
            recalls[mode].append(recall)

            if SAVE_PLOTS or SHOW_PLOTS:
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

        plt.close('all')

# print(IOUs)

for key, val in IOUs.items():
    IOU_avg[key] = np.average(val)

print("----- IOU -----\n", IOU_avg)

for key, val in precisions.items():
    precision_avg[key] = np.average(val)

print("----- Precision -----\n", precision_avg)

for key, val in recalls.items():
    recall_avg[key] = np.average(val)

print("----- Recall -----\n", recall_avg)




