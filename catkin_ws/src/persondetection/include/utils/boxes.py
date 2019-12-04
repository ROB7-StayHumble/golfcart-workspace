import cv2
import glob
import numpy as np

from connected_components.connectedcomponents import calculate_confidence_score
class Box:
    def __init__(self, img, xyxy = None, center = None, size = None, confidence=None):
        self.img_h, self.img_w = img.shape[:2]
        self.img = img
        if len(img.shape) == 2:
            self.color_img = cv2.cvtColor(self.img,cv2.COLOR_GRAY2BGR)
        else:
            self.color_img = self.img
        if xyxy == None:
            if size == None:
                self.h=np.random.randint(0,self.img_h)
                self.w=np.random.randint(0,self.img_w)
            else:
                self.h = size['height']
                self.w = size['width']
            if center == None:
                x_min = 0+self.w/2
                y_min = 0+self.h/2
                self.x=np.random.randint(x_min,self.img_w-self.w/2)
                self.y=np.random.randint(y_min,self.img_h-self.h/2)
            else:
                self.x = center['x']
                self.y = center['y']
            self.top_left = {  'x':int(self.x - self.w/2),
                                'y':int(self.y - self.y/2)}
            self.bottom_right = {   'x':int(self.x + self.w/2),
                                    'y':int(self.y + self.y/2)}
            self.xyxy = [self.top_left['x'],self.top_left['y'],self.bottom_right['x'],self.bottom_right['y']]
        else:
            self.xyxy = xyxy
            self.top_left = {'x':xyxy[0],
                             'y':xyxy[1]}
            self.bottom_right = {'x':xyxy[2],
                                 'y':xyxy[3]}
            self.h = xyxy[3] - xyxy[1]
            self.w = xyxy[2] - xyxy[0]
            self.x = xyxy[0] + self.w/2
            self.y = xyxy[1] + self.h/2

        self.angle = angle_from_box(self.img,self.xyxy)
        if confidence == None:
            self.confidence = np.random.randint(0,100)/100
        else: self.confidence = confidence

        rel_h = self.h/self.img_h
        rel_w = self.w/self.img_w
        rel_y = self.y/self.img_h
        rel_x = self.x/self.img_w

        score_yh = np.power(calculate_confidence_score(rel_y, rel_h, SLOPE['y_height'], INTERCEPT['y_height']),2)
        score_aspect_ratio = np.power(calculate_confidence_score(rel_h, rel_w, SLOPE['aspect_ratio'], INTERCEPT['aspect_ratio']), 2)
        score_dimensions = np.round(np.round(score_yh, decimals=2) * np.round(score_aspect_ratio, decimals=2),decimals=2)
        total_score = score_dimensions

        # print(total_score)
        self.score = np.round(total_score,decimals=2)
        print(self.score)
    def drawOnImg(self, img = None):
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        except:
            img = self.color_img
        box_color = (0,255,0)
        img = cv2.rectangle(img,(self.top_left['x'],self.top_left['y']),(self.bottom_right['x'],self.bottom_right['y']),box_color,3)
        img = cv2.circle(img,(self.x,self.y), int(self.img_h/50), (0,0,255), -1)
        img = cv2.putText(img, str(np.round(self.confidence,decimals=2)), (self.top_left['x']-10, self.top_left['y']-10), cv2.FONT_HERSHEY_PLAIN, 1.0, box_color, 2)
        return img
    def makeConfidenceMap(self):
        map = np.zeros_like(self.img)
        map[self.top_left['y']:self.bottom_right['y'],
            self.top_left['x']:self.bottom_right['x']] = 255
        return map

def makeConfidenceMapFromBoxes(img,boxes):
    map = np.float64(np.zeros_like(img))
    boxes_sorted = sorted(boxes,key=lambda box: box.score)
    for box in boxes_sorted:
        if box.score > 0.5:
            map[box.top_left['y']:box.bottom_right['y'],
                box.top_left['x']:box.bottom_right['x']] = box.score
    return map


# file = glob.glob("*.png")[0]
# img = cv2.imread(file,0)
# box1 = Box(img)
# box2 = Box(img)
# img_boxes = box1.drawOnImg(img)
# img_boxes = box2.drawOnImg(img_boxes)
#
# map = makeConfidenceMapFromBoxes(img,[box1,box2])
#
# cv2.namedWindow('boxes',cv2.WINDOW_NORMAL)
# cv2.namedWindow('confidence',cv2.WINDOW_NORMAL)
#
# cv2.imshow('boxes',img_boxes)
# #confidence_map = box1.makeConfidenceMap()
# #normalized_map = cv2.normalize(map, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow('confidence',map)
# cv2.waitKey(0)