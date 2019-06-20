import cv2
import numpy as np
import sys
import os
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

img_root_folder = './dataset/originalPics'
img_save_folder = './dataset/cropPicsNeg'
name_list_folder = './dataset/FDDB-folds'
name_list_save_folder = './dataset/cropPicsNeg/list'

def add_anno_on_img(img, annos):
    new_img = img.copy()
    for anno in annos:
        center = (int(anno[3]), int(anno[4]))
        axes = (int(anno[1]), int(anno[0]))
        angle = anno[2]
        width = 2*axes[0] * 4 // 3
        height = 2*axes[1] * 4 //3 
        cv2.ellipse(new_img, center, axes, -angle, 0, 360, (255, 0, 0), 3)
        cv2.rectangle(new_img, (center[0] - width//2, center[1] - height//2), (center[0] + width//2, center[1] + height//2), (0, 255, 0), 3)
    cv2.imshow('img', new_img)
    cv2.waitKey()

def crop_img(img, annos, img_name):
    height_ori, width_ori = img.shape[0], img.shape[1]
    for i, anno in enumerate(annos):
        center_x, center_y = int(anno[3]), int(anno[4])
        width, height = 2 * int(anno[1]) * 4 // 3, 2 * int(anno[0]) * 4 // 3
        # print('width={}, height={}'.format(width, height))
        j = 0
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                if(x_offset == 0 and y_offset == 0):
                    continue

                left = center_x - width//2 + x_offset*width//3
                right = center_x + width//2 + x_offset*width//3
                top = center_y - height//2 + y_offset*height//3
                bottom = center_y + height//2 + y_offset*height//3
                # print('l={}, r={}, t={}, b={}'.format(left, right, top, bottom))
                # print(max(0, top), min(img.shape[0]-1, bottom), max(0, left), min(img.shape[1]-1, right))
                img_crop = img[max(0, top):min(height_ori, bottom), max(0, left):min(width_ori, right)]
                # print(max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width_ori))
                img_crop = cv2.copyMakeBorder(img_crop, max(0, -top), max(0, bottom - height_ori), max(0, -left), max(0, right - width_ori), cv2.BORDER_REPLICATE)

                full_path = os.path.join(img_save_folder, img_name + '_{0}_{1}.jpg'.format(i, j))
                dir_path, _ = os.path.split(full_path)
                if(not os.path.exists(dir_path)):
                    os.makedirs(dir_path)
                cv2.imwrite(full_path, img_crop)
                j += 1

def random_crop_img(img, annos, img_name):
    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        l = max(x1-w1//2, x2-w2//2)
        r = min(x1+w1//2, x2+w2//2)
        t = max(y1-h1//2, y2-h2//2)
        b = min(y1+h1//2, y2+h2//2)

        inter = max(0, r - l) * max(0, b - t)

        return inter / (w1*h1 + w2*h2 - inter)

    height_ori, width_ori = img.shape[0], img.shape[1]
    for i, anno in enumerate(annos):
        center_x, center_y = int(anno[3]), int(anno[4])
        width, height = 2 * int(anno[1]) * 4 // 3, 2 * int(anno[0]) * 4 // 3
        j = 0
        while j < 80:
            w = random.randint(20, height_ori)
            h = random.randint(20, width_ori)
            if not 0.6 < w / h < (1/0.6):
                continue
            x = random.randint(0, width)
            y = random.randint(0, height)
            # print(j, x, y, w, h)
            
            overlap = False
            for a in annos:
                cx, cy = int(a[3]), int(a[4])
                aw, ah = 2 * int(a[1]) * 4 //3, 2 * int(a[0]) * 4 //3
                if iou((x,y,w,h), (cx,cy,aw,ah)) > 0.4:
                    overlap = True
                    break
            if overlap:
                continue

            left = x - w//2
            right = x + w//2
            top = y - h//2
            bottom = y + h//2

            img_crop = img[max(0, top):min(height_ori, bottom), max(0, left):min(width_ori, right)]
            img_crop = cv2.copyMakeBorder(img_crop, max(0, -top), max(0, bottom - height_ori), max(0, -left), max(0, right - width_ori), cv2.BORDER_REPLICATE)

            # cv2.imshow('random crop', img_crop)
            # cv2.waitKey()

            full_path = os.path.join(img_save_folder, img_name + '_{0}_{1}.jpg'.format(i, j+8))
            dir_path, _ = os.path.split(full_path)
            if(not os.path.exists(dir_path)):
                os.makedirs(dir_path)
            cv2.imwrite(full_path, img_crop)
            j += 1

def process_one_list(anno_txt_name, list_txt_name):
    print(anno_txt_name, list_txt_name)
    f_anno =  open(os.path.join(name_list_folder, anno_txt_name), 'r')
    f_list = open(os.path.join(name_list_save_folder, list_txt_name), 'w')

    while(True):
        img_name = f_anno.readline().strip()
        print(img_name)
        img_path = img_name + '.jpg'
        img = cv2.imread(os.path.join(img_root_folder, img_path))
        if(img_name == ''):
            break
        annos_num = int(f_anno.readline())
        annos = []
        for i in range(annos_num):
            annos.append([float(x) for x in f_anno.readline().split()])
            for j in range(8):
                f_list.write(img_name + '_{0}_{1}\n'.format(i, j))
        # add_anno_on_img(img, annos)
        crop_img(img, annos, img_name)
        # random_crop_img(img, annos, img_name)

    f_anno.close()
    f_list.close()

for i_list in range(10):
    if(i_list >= 4 and i_list < 9):
        continue
    anno_txt_name = 'FDDB-fold-%02d-ellipseList.txt' % (i_list + 1)
    list_txt_name = 'FDDB-fold-%02d.txt' % (i_list + 1)
    process_one_list(anno_txt_name, list_txt_name)
    print('{}% has completed!'.format(10*(i_list+1)))
    print('-----------------------------------------')
    # break
