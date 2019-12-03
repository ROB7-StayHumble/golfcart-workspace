#!/usr/bin/env python3
from utils.boxes import *
from utils.transformbox import get_boxes_zedframe
from connected_components.connectedcomponents import *
import matplotlib.pyplot as plt

folder = "src/persondetection/src/testing/"
ir = cv2.imread(folder+"test_img/ircam1571825077852250111.png")
zed = cv2.imread(folder+"test_img/zedcam1571825077852250111.png")

boxes_ir = [{'coords':[206,194,276,372],'conf':1}]
boxes_ir_class = [Box(ir,xyxy=box['coords'],confidence=box['conf']) for box in boxes_ir]
boxes_tf = get_boxes_zedframe(boxes_ir_class)

for box in boxes_ir_class:
    x1, y1, x2, y2  = box.top_left['x'],box.top_left['y'],box.bottom_right['x'],box.bottom_right['y']
    cv2.rectangle(ir,(x1,y1),(x2,y2),(0,0,255),2)

for box in boxes_tf:
    x1, y1, x2, y2  = box.top_left['x'],box.top_left['y'],box.bottom_right['x'],box.bottom_right['y']
    cv2.rectangle(zed,(x1,y1),(x2,y2),(0,0,255),2)

images = [ir,zed]
plt.figure(num=None, figsize=(12, 4), dpi=300)

for i in range(len(images)):
    plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
    # plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


plt.savefig(folder+"results/transformbox/tf.png", bbox_inches='tight')
plt.show()