import glob
import cv2
import numpy as np
from random import Random

SLOPE = dict()
INTERCEPT = dict()
SLOPE['y_height'] = 1.838553092116205
INTERCEPT['y_height'] = -0.592323173694313
SLOPE['aspect_ratio'] = 0.2196364045520636
INTERCEPT['aspect_ratio'] = 0.004519400085200226

# # ZED
# SLOPE['y_height'] = 1.838772993923768
# INTERCEPT['y_height'] = -0.592406200468761
# SLOPE['aspect_ratio'] = 0.2193557959956911
# INTERCEPT['aspect_ratio'] = 0.003921820805625677

def distance_from_line(x,y,a,b):
    predicted_y = x*a + b
    diff = np.abs(y - predicted_y)
    return diff

def calculate_confidence_score(box_x,box_y,model):
    model_slope = SLOPE[model]
    model_intercept = INTERCEPT[model]

    diff_max = np.max([ distance_from_line(0,0,model_slope,model_intercept),
                        distance_from_line(0,1,model_slope,model_intercept),
                        distance_from_line(1,1,model_slope,model_intercept),
                        distance_from_line(1,0,model_slope,model_intercept)])
    diff = distance_from_line(box_x,box_y,model_slope,model_intercept)
    score = np.power(1 - diff/(diff_max),2)

    # score = 1 - diff / (diff_max)
    #print(diff)
    #print(box_x,box_y,model_slope,model_intercept,score)
    return np.round(score,decimals=2)

def combine_confidence_scores(scores):
    total_score = 1
    for score in scores:
        total_score *= score
    return total_score


def detect_connected_components_updated(img):
    orig_img = img.copy()
    img_h, img_w = orig_img.shape[:2]
    img = orig_img.copy()
    # img = img[:,:,0]

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    blur = cv2.GaussianBlur(img, (21, 21), 3)

    grad_x = cv2.Sobel(blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad_binary = grad.copy()
    grad_binary[grad > 15] = 255
    grad_binary[grad <= 15] = 0

    # grads = [orig_img, blur, grad, grad_binary]
    #
    # plt.figure(num=None, figsize=(12, 4), dpi=300)
    # for i in range(len(grads)):
    #     plt.subplot(1, len(grads), i + 1), plt.imshow(grads[i], 'gray')
    #     # plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    #
    # plt.savefig(folder + "results/sobel/" + imgpath.split("/")[-1], bbox_inches='tight')

    avgPixelIntensity = cv2.mean(img)
    # print("Average intensity of image: ", avgPixelIntensity[0])
    avg = avgPixelIntensity[0]
    # thresh = avg + 0.9*avg
    thresh = avg * 1.14

    img[img > thresh] = 255
    img[img <= thresh] = 0
    img = 255 - img

    # based on https://stackoverflow.com/questions/40777826/detect-black-ink-blob-on-paper-opencv-android

    def random_color(random):
        """
        Return a random color
        """
        icolor = random.randint(0, 0xFFFFFF)
        return [icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff]

    # Read as Grayscale
    if len(img.shape) == 2:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        cimg = img

    # bilateralFilter to remove noisy region, comment to see its affect.

    mask = 255 - img

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_8U)

    # cv2.imwrite("3-connectedcomponent.jpg", labels)

    # create the random number
    random = Random()
    boxes = []

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # if h > 50 and top + h > 200:
        # print(total_score)
        box_color = (random_color(random))
        boxes.append((left, top, left + w, top + h))
        cv2.rectangle(cimg, (left, top), (left + w, top + h), box_color, 2)

    grad_binary = 255 - grad_binary
    cgrad = cv2.cvtColor(grad_binary, cv2.COLOR_GRAY2BGR)
    mask = 255 - grad_binary

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_8U)

    # cv2.imwrite("3-connectedcomponent.jpg", labels)

    # create the random number
    random = Random()

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # if h > 50 and top + h > 200:
        # print(total_score)
        box_color = (random_color(random))
        boxes.append((left, top, left + w, top + h))
        cv2.rectangle(cgrad, (left, top), (left + w, top + h), box_color, 2)

    # boxes = non_max_suppression_fast(np.array(boxes), 0.5)

    # for box in boxes:
    #     box_color = (random_color(random))
    #     cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), box_color, 2)

    # images = [orig_img, cimg, cgrad]
    #
    # plt.figure(num=None, figsize=(12, 4), dpi=300)
    #
    # for i in range(len(images)):
    #     plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
    #     # plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    #
    # # plt.show()
    # plt.savefig(folder + "results/connectedcomp/" + imgpath.split("/")[-1], bbox_inches='tight')

    return cimg, cgrad, boxes


def detect_connected_components(orig_img):
    img_h, img_w = orig_img.shape[:2]
    img = orig_img
    if len(orig_img.shape) > 2:
        img = img[:,:,0]

    avgPixelIntensity = cv2.mean( img )
    # print("Average intensity of image: ", avgPixelIntensity[0])
    avg = avgPixelIntensity[0]
    thresh = avg + 0.25*avg
    # thresh = int(avg + 0.25*avg)

    img[img > thresh] = 255
    img[img <= thresh] = 0
    img = 255 - img

    img = cv2.bilateralFilter(img, 5, 3, 10)

    #
    #Find average intensity to distinguish paper region
    # avgPixelIntensity = cv2.mean( img )
    # print("Average intensity of image: ", avgPixelIntensity[0])
    # #thresh = 150
    # avg = avgPixelIntensity[0]
    # #thresh = avg + 0.9*avg
    # thresh = avg
    #
    # img[img > thresh] = 255
    # img[img <= thresh] = 0

    #based on https://stackoverflow.com/questions/40777826/detect-black-ink-blob-on-paper-opencv-android

    def random_color(random):
        """
        Return a random color
        """
        icolor = random.randint(0, 0xFFFFFF)
        return [icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff]



    # bilateralFilter to remove noisy region, comment to see its affect.


    mask = 255 - img

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_8U)

    #cv2.imwrite("3-connectedcomponent.jpg", labels)

    # create the random number
    random = Random()
    boxes = []

    #Read as Grayscale
    if len(img.shape) == 2:
        cimg = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    else: cimg = mask

    for i in range(1, num_labels):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # rel_h = h/img_h
        # rel_w = w/img_w
        # rel_y = (top + h/2)/img_h
        # rel_x = (left + w/2)/img_w

        # score_yh = calculate_confidence_score(rel_y,rel_h,SLOPE['y_height'],INTERCEPT['y_height'])
        # score_aspect_ratio = calculate_confidence_score(rel_h,rel_w,SLOPE['aspect_ratio'],INTERCEPT['aspect_ratio'])
        # total_score = np.mean([score_yh,score_aspect_ratio])
        # total_score = score_yh
        if h > 50 and top + h > 200:
        # print(total_score)
            box_color = (random_color(random))
            boxes.append({'coords':[left, top, left + w, top + h],
                         'conf':1})
            cv2.rectangle(cimg, (left, top), (left + w, top + h), box_color, 2)
            # cv2.putText(cimg, str(np.round(total_score,decimals=2)), (left-10, top-10), cv2.FONT_HERSHEY_PLAIN, 1.0, box_color, 1)

    return cimg, boxes

# orig = cv2.imread('/home/zacefron/Desktop/golfcart-sensorfusion/sensorfusion/cam_data/ir/ircam1571746634692805517.png',0)
# cimg,boxes = detect_connected_components(orig.copy())
# cv2.imshow('orig',orig)
# cv2.imshow('cimg',cimg)
# cv2.waitKey(0)