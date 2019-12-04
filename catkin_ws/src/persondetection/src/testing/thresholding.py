#!/usr/bin/env python3
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import Random

folder = "src/persondetection/src/testing/"

masks = {
         'ircam1571825132561437576.png':[200,385, 550,615],
         'ircam1571825294694103257.png':[200,565, 500,642],
         'ircam1571746715348190652.png':[195,405,600,670],
         'ircam1571746625354059903.png':[170,245,415,445],
         'ircam1571825078917436229.png':[195,400,385,460],
         '1571825186737317588_IR.png':[180,265,580,625],
         '1571825142073643588_IR.png':[200,480,0,93]
         }


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
         
def detect_connected_components(imgpath):
    orig_img = cv2.imread(imgpath,0)
    img_h, img_w = orig_img.shape[:2]
    img = orig_img.copy()
    #img = img[:,:,0]
    
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    blur = cv2.GaussianBlur(img,(21,21),3)
    
    grad_x = cv2.Sobel(blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad_binary = grad.copy()
    grad_binary[grad > 15] = 255
    grad_binary[grad <= 15] = 0

    grads = [orig_img,blur,grad,grad_binary]

    plt.figure(num=None, figsize=(12, 4), dpi=300)
    for i in range(len(grads)):
        plt.subplot(1, len(grads), i + 1), plt.imshow(grads[i], 'gray')
        #plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.savefig(folder+"results/sobel/"+imgpath.split("/")[-1], bbox_inches='tight')

    avgPixelIntensity = cv2.mean( img )
    # print("Average intensity of image: ", avgPixelIntensity[0])
    avg = avgPixelIntensity[0]
    #thresh = avg + 0.9*avg
    thresh = avg * 1.14

    img[img > thresh] = 255
    img[img <= thresh] = 0
    img = 255 - img

    img = cv2.bilateralFilter(img,5, 3, 10)

    #based on https://stackoverflow.com/questions/40777826/detect-black-ink-blob-on-paper-opencv-android

    def random_color(random):
        """
        Return a random color
        """
        icolor = random.randint(0, 0xFFFFFF)
        return [icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff]

    #Read as Grayscale
    if len(img.shape) == 2:
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    else: cimg = img

    # bilateralFilter to remove noisy region, comment to see its affect.


    mask = 255 - img

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_8U)

    #cv2.imwrite("3-connectedcomponent.jpg", labels)

    # create the random number
    random = Random()
    boxes = []

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        #if h > 50 and top + h > 200:
        # print(total_score)
        box_color = (random_color(random))
        boxes.append((left, top, left + w, top + h))
        cv2.rectangle(cimg, (left, top), (left + w, top + h), box_color, 2)
        
    grad_binary = 255 - grad_binary
    cgrad = cv2.cvtColor(grad_binary,cv2.COLOR_GRAY2BGR)
    mask = 255 - grad_binary

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_8U)

    #cv2.imwrite("3-connectedcomponent.jpg", labels)

    # create the random number
    random = Random()

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        #if h > 50 and top + h > 200:
        # print(total_score)
        box_color = (random_color(random))
        boxes.append((left, top, left + w, top + h))
        cv2.rectangle(cgrad, (left, top), (left + w, top + h), box_color, 2)

    # boxes = non_max_suppression_fast(np.array(boxes), 0.5)

    # for box in boxes:
    #     box_color = (random_color(random))
    #     cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), box_color, 2)
            
    images = [orig_img, cimg, cgrad]

    plt.figure(num=None, figsize=(12, 4), dpi=300)

    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
        #plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        
    # plt.show()
    plt.savefig(folder+"results/connectedcomp/"+imgpath.split("/")[-1], bbox_inches='tight')

    return cimg, boxes

def thresholding(imgpath):
    img = cv2.imread(imgpath)
    lo = np.mean(img) * 1.14
    print(lo)
    hi = 255
    # ret, thresh1 = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY)
    ret, thresh_bin = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(img, lo, hi, cv2.THRESH_TRUNC)
    ret, thresh1 = cv2.threshold(img, lo, hi, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(img, lo, hi, cv2.THRESH_TOZERO_INV)

    lo = np.mean(thresh1)*4
    print(lo)
    ret, thresh2 = cv2.threshold(thresh1, lo, hi, cv2.THRESH_TOZERO)
    titles = ['Original Image', 'TOZERO', 'TOZERO']

    images = [img, thresh1, thresh2]

    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    # plt.show()

    return img, thresh_bin

def histo(imgpath):
    print(imgpath)
    img = cv2.imread(imgpath)
    # create a mask

    mask = np.zeros(img.shape[:2], np.uint8)
    y1,y2,x1,x2 = masks[imgpath][0],masks[imgpath][1],masks[imgpath][2],masks[imgpath][3]
    mask[y1:y2, x1:x2] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    hist_full_norm = cv2.normalize(hist_full, hist_full, 1, 0, cv2.NORM_L1)

    hist_mask_norm = cv2.normalize(hist_mask, hist_mask, 1, 0, cv2.NORM_L1)

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full_norm), plt.plot(hist_mask_norm)
    plt.axvline(img.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.xlim([0, 256])
    min_ylim, max_ylim = plt.ylim()
    plt.text(img.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(img.mean()))

    plt.show()
    return img


for imgpath in glob.glob(folder+"test_img/*.png"):
    #thresholding(imgpath)
    print(imgpath)
    detect_connected_components(imgpath)
    # histo(imgpath)
