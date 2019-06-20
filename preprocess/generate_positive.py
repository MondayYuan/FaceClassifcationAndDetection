import cv2
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

img_root_folder = './dataset/originalPics'
img_save_folder = './dataset/cropPicsPos'
name_list_folder = './dataset/FDDB-folds'
name_list_save_folder = './dataset/cropPicsPos/list'


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
        left = center_x - width//2
        right = center_x + width//2
        top = center_y - height//2
        bottom = center_y + height//2
        # print('l={}, r={}, t={}, b={}'.format(left, right, top, bottom))
        # print(max(0, top), min(img.shape[0]-1, bottom), max(0, left), min(img.shape[1]-1, right))
        img_crop = img[max(0, top):min(height_ori, bottom), max(0, left):min(width_ori, right)]
        # print(max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width_ori))
        img_crop = cv2.copyMakeBorder(img_crop, max(0, -top), max(0, bottom - height_ori), max(0, -left), max(0, right - width_ori), cv2.BORDER_REPLICATE)

        full_path = os.path.join(img_save_folder, img_name + '_{0}.jpg'.format(i))
        dir_path, _ = os.path.split(full_path)
        if(not os.path.exists(dir_path)):
            os.makedirs(dir_path)
        cv2.imwrite(full_path, img_crop)


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
            f_list.write(img_name + '_{}\n'.format(i))
        # add_anno_on_img(img, annos)
        crop_img(img, annos, img_name)

    f_anno.close()
    f_list.close()

for i_list in range(10):
    anno_txt_name = 'FDDB-fold-%02d-ellipseList.txt' % (i_list + 1)
    list_txt_name = 'FDDB-fold-%02d.txt' % (i_list + 1)
    process_one_list(anno_txt_name, list_txt_name)
    print('{}% has completed!'.format(10*(i_list+1)))
    print('-----------------------------------------')
    # break
