import numpy as np
import cv2
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from classify.logistic_model import LogisticModel
from classify.cnn_model import CNNModel
from classify.svm_model import SVMModel

MIN_WIN = 30
WIN_SLIDE = 3
WIN_RATIO = 1.333

MIN_AREA = 100
MIN_RATIO = 0.6
RATIO_STEP = 0.1

IOU_THRESH = 0.4
N_TOP = 30

class Detector(object):
    def __init__(self, classifier):
        self.classfier = classifier
    
    def detect(self, img):
        boxes_list = []
        imgs_list = []
        # new_img = np.copy(img)
        start = time.time()
        for box in self.sliding_windows(img.shape[0], img.shape[1]):
            x, y, w_window, h_window = box
            crop_img = img[y:min(img.shape[0], y+h_window), x:min(img.shape[1], x+w_window)]
            if crop_img.shape[0] * crop_img.shape[1] < MIN_AREA or crop_img.shape[0] < MIN_WIN or crop_img.shape[1] < MIN_WIN:
                continue
            boxes_list.append(box)
            imgs_list.append(crop_img)
            # cv2.rectangle(new_img, (x, y), (x+w_window, y+h_window), (0, 0, 255))
        print('generate boxes time = {}'.format(time.time() - start))
        print('n_boxes = {}'.format(len(boxes_list)))
    
        start = time.time()
        p = self.classfier.predict(imgs_list)
        print('pretict time = {}'.format(time.time() - start))

        start = time.time()
        idx = np.argsort(p)[-N_TOP:][::-1]
        p, boxes_list = p[idx], np.array(boxes_list)[idx]
        idx = (p>0.5)
        p, boxes_list = p[idx], [tuple(box) for box in boxes_list[idx]]
        p_boxes = self.nms(p, boxes_list)
        print('n_boxes =', len(p_boxes))
        print('nms time = {}'.format(time.time() - start))
        
        for p, box in p_boxes:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255))
            cv2.putText(img, str(p), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        
        # cv2.imshow('img', img)
        # cv2.waitKey()
        
        return img

        # return p
    def nms(self, ps, boxes):
        p_boxes = list(zip(ps, boxes))
        for i, (p, box) in enumerate(p_boxes):
            for p_box in p_boxes[i+1:]:
                box2 = p_box[1]
                if(self.iou(box, box2) > IOU_THRESH):
                    p_boxes.remove(p_box)
        return p_boxes
    
    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        l = max(x1, x2)
        r = min(x1+w1, x2+w2)
        t = max(y1, y2)
        b = min(y1+h1, y2+h2)

        # cover totally
        if (l, t, r-l, b-t) in [box1, box2]:
            return 1.0 

        inter = max(0, r - l) * max(0, b - t)

        return inter / (w1*h1 + w2*h2 - inter)

    def sliding_windows(self, h_img, w_img):
        boxes = []
        w_window = MIN_WIN
        while w_window < w_img:
            for ratio in np.arange(MIN_RATIO, 1.0001, RATIO_STEP):
                h_list = [int(w_window * ratio), int(w_window / ratio)]
                if(abs(h_list[0] - h_list[1]) < 5):
                    h_list = h_list[0:1]
                for h_window in h_list:
                    if w_window * h_window < MIN_AREA or h_window < MIN_WIN:
                        continue
                    for x in range(0, w_img, max(w_window//WIN_SLIDE, 5)):
                        for y in range(0, h_img, max(h_window//WIN_SLIDE, 5)):
                            boxes.append((x, y, w_window, h_window))
            # break
            w_window = int(w_window * WIN_RATIO)
        return boxes
                            


if __name__ == '__main__':
    classfier = LogisticModel(use_pretrained=True)
    # classfier = CNNModel(use_pretrained=True)
    # classfier = SVMModel(use_pretrained=True, model_save_path='./classify/pre_train/svm_linear.pkl')
    detector = Detector(classfier)
    
    img = cv2.imread('dataset/originalPics/2002/07/21/big/img_96.jpg')
    # start = time.time()
    detector.detect(img)
    # print(time.time() - start)
