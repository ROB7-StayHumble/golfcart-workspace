#!/usr/bin/env python3

from utils.boxes import *
from utils.img_utils import *
from utils.transformbox import get_boxes_zedframe
from persondetection.run_yolo import *
from connected_components.connectedcomponents import *

SHOW_PLOTS = False
SAVE_PLOTS = True
cam = "IR"

folder = "src/persondetection/src/testing/"

pairs = [
         # ("1571746625368332498_ircam.png","1571746625385033721_zed.png","1571746625385033721_GT.png"),
         # ("1571746631630210616_ircam.png","1571746631605020001_zed.png","1571746631605020001_GT.png"),
         # ("1571746633060068411_ircam.png","1571746633070918853_zed.png","1571746633070918853_GT.png"),
         ("1571746634460540239_ircam.png","1571746634504400900_zed.png","1571746634504400900_GT.png")]

# ir_images, zed_images, gt_images = [], [], []
# for tup in pairs:
#     ir_images.append(tup[0])
#     zed_images.append(tup[1])
#     gt_images.append(tup[2])

# pairs = list(zip(ir_images, zed_images, gt_images))

IOUs = {}
IOU_avg = {}

precisions = {}
precision_avg = {}

recalls = {}
recall_avg = {}

modes = ["combo sum"]
thresholds = np.linspace(0,1,num=10,endpoint=False)

for mode in modes:
    IOUs[mode] = {}
    precisions[mode] = {}
    recalls[mode] = {}
    IOU_avg[mode] = {}
    precision_avg[mode] = {}
    recall_avg[mode] = {}
    for threshold in thresholds:
        IOUs[mode][str(threshold)] = []
        precisions[mode][str(threshold)] = []
        recalls[mode][str(threshold)] = []
        IOU_avg[mode][str(threshold)] = []
        precision_avg[mode][str(threshold)] = []
        recall_avg[mode][str(threshold)] = []

for i,pair in enumerate(pairs):
    if i < 500:
        print(pair)
        ir = cv2.imread(folder+"test_img/ir/"+pair[0])
        ir_1channel = cv2.imread(folder + "test_img/ir/" + pair[0],0)
        zed = cv2.imread(folder + "test_img/zed/" + pair[1])

        gt = cv2.imread(folder + "test_img/gt/" + pair[2])

        ir_cc_img_thresh, ir_cc_img_grad, ir_cc_boxes = detect_connected_components_updated(ir_1channel)
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

        filename = pair[1].split(".")[-2]

        for mode in modes:

            if mode.startswith("combo"):
                # total_map = (ir_yolo_map + ir_cc_map + zed_yolo_map)/3
                # ir_yolo_map[np.bitwise_and(ir_cc_map > 0,ir_yolo_map > 0)] = 1
                total_map_max = np.maximum.reduce([ir_yolo_map, ir_cc_map, zed_yolo_map])
                total_map_sum = (ir_yolo_map + ir_cc_map + zed_yolo_map)/3

                if mode == "combo max":
                    total_map = total_map_max
                elif mode == "combo sum":
                    total_map = total_map_sum

                zed_yolo_img = cv2.cvtColor(zed_yolo_img, cv2.COLOR_BGR2RGB)

                # if SAVE_PLOTS or SHOW_PLOTS:
                #     images = [ir_cc_img_thresh,ir_yolo_img,zed_yolo_img,
                #               ir_cc_map,ir_yolo_map,zed_yolo_map]
                #     titles = ["Connected components on IR image","YOLOv3-tiny on IR image", "YOLOv3-tiny on RGB image"]
                #     fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
                #
                #     plt.rcParams["axes.titlesize"] = 6
                #     for i in range(len(images)):
                #         plt.subplot(2, len(images)/2, i + 1), plt.imshow(images[i], 'gray')
                #         if i < len(titles):
                #             plt.title(titles[i])
                #         plt.xticks([]), plt.yticks([])
                #

                #     plt.savefig(folder + "results/total_confmap/" + filename + "_detail.png", bbox_inches='tight')
                #
                #     images = [total_map_sum,total_map_max]
                #     titles = ["Total confidence map\n(Average approach)","Total confidence map\n(Element-wise maximum approach)"]
                #     fig = plt.figure(num=None, figsize=(12, 4), dpi=300)
                #
                #     plt.rcParams["axes.titlesize"] = 8
                #     for i in range(len(images)):
                #         plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
                #         plt.title(titles[i])
                #         plt.xticks([]), plt.yticks([])
                #
                #     plt.savefig(folder + "results/total_confmap/"+filename+"_total.png", bbox_inches='tight')
                #     # plt.show()

                # threshold = 0.45

            elif mode == "ir+zed":
                total_map_max = np.maximum.reduce([ir_yolo_map, zed_yolo_map])
                total_map_sum = (ir_yolo_map + zed_yolo_map)/2
                total_map = total_map_max
                # threshold = 0

            elif mode == "zed":
                total_map = zed_yolo_map
                # threshold = 0

            elif mode == "ir":
                total_map = ir_yolo_map
                # threshold = 0

            else: continue

            cv2.imwrite(
                folder + "results/eval_illustration/" + filename + "_" + mode + ".png",
                cv2.convertScaleAbs(total_map, alpha=(255.0)))

            for threshold in thresholds:
                total_map_bin = np.float32(total_map)
                total_map_bin[np.float32(total_map) > threshold] = 1
                total_map_bin[np.float32(total_map) <= threshold] = 0
                gt_intersection = np.bitwise_and(total_map_bin > 0, gt > 0)
                gt_union = np.bitwise_or(total_map_bin > 0, gt > 0)
                iou = np.sum(gt_intersection)/np.sum(gt_union)

                TP = np.sum(gt_intersection)
                FP = np.sum(np.bitwise_and(total_map_bin > 0, gt == 0))
                FN = np.sum(np.bitwise_and(total_map_bin == 0, gt > 0))
                if TP+FP > 0:
                    precision = TP/(TP+FP)
                else:
                    precision = 0
                recall = TP/(TP+FN)

                # print(threshold,recall)
                IOUs[mode][str(threshold)].append(iou)
                precisions[mode][str(threshold)].append(precision)
                recalls[mode][str(threshold)].append(recall)

                if SAVE_PLOTS or SHOW_PLOTS:
                    images = [gt, total_map_bin, np.float32(gt_intersection)]

                    cv2.imwrite(folder + "results/eval_illustration/" + filename + "_" + mode + "_gt.png", gt)
                    cv2.imwrite(
                        folder + "results/eval_illustration/" + filename + "_" + mode + "_" + str(threshold) + ".png",
                        cv2.convertScaleAbs(total_map_bin, alpha=(255.0)))
                    cv2.imwrite(
                        folder + "results/eval_illustration/" + filename + "_" + mode + "_" + str(threshold) + "_intersection.png",
                        cv2.convertScaleAbs(np.float32(gt_intersection), alpha=(255.0)))
                    # plt.show()

        plt.close('all')

# print(IOUs)



data = []

for mode in modes:
    for threshold in thresholds:
        precision_avg[mode][str(threshold)] = np.average(precisions[mode][str(threshold)])
        IOU_avg[mode][str(threshold)] = np.average(IOUs[mode][str(threshold)])
        recall_avg[mode][str(threshold)] = np.average(recalls[mode][str(threshold)])
        data.append({'precision': precision_avg[mode][str(threshold)],
         'recall': recall_avg[mode][str(threshold)],
         'label': mode})

F1s = {}
F1s_max = {}
for mode in modes:
    F1s[mode] = {}
    for threshold in thresholds:
        p = precision_avg[mode][str(threshold)]
        r = recall_avg[mode][str(threshold)]
        if (p+r) > 0:
            F1s[mode][str(threshold)] = (2*p*r)/(p+r)
        else: F1s[mode][str(threshold)] = 0
    F1s_max[mode] = np.max(np.array(list(F1s[mode].values())))
    max_thresh = max(F1s[mode], key=F1s[mode].get)

    print(mode, "max F1:", F1s_max[mode], "thresh", max_thresh)
    print(mode, "---> precision", precision_avg[mode][max_thresh])
    print(mode, "---> recall", recall_avg[mode][max_thresh])
    print(mode, "---> IoU", IOU_avg[mode][max_thresh])


# print(IOU_avg['combo'],precision_avg['combo'],recall_avg['combo'])

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# df = pd.DataFrame(data)
# # print(df)
#
# sns.lineplot(x="recall", y="precision", hue="label", data=df)
# plt.show()


# print("----- IOU -----\n", IOU_avg)
#
# for key, val in precisions['combo']['0.4'].items():
#     precision_avg[key] = np.average(val)
#
# print("----- Precision -----\n", precision_avg)
#
# for key, val in recalls['combo']['0.4'].items():
#     recall_avg[key] = np.average(val)
#
# print("----- Recall -----\n", recall_avg)




