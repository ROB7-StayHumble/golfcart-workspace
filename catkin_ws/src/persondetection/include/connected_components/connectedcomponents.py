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

def calculate_confidence_score(box_x,box_y,model_slope=SLOPE['y_height'],model_intercept=INTERCEPT['y_height']):

    diff_max = np.max([ distance_from_line(0,0,model_slope,model_intercept),
                        distance_from_line(0,1,model_slope,model_intercept),
                        distance_from_line(1,1,model_slope,model_intercept),
                        distance_from_line(1,0,model_slope,model_intercept)])
    diff = distance_from_line(box_x,box_y,model_slope,model_intercept)
    score = np.power(1 - diff/(diff_max),2)
    #print(diff)
    #print(box_x,box_y,model_slope,model_intercept,score)
    return np.round(score,decimals=2)

def combine_confidence_scores(scores):
    total_score = 1
    for score in scores:
        total_score *= score
    return total_score

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