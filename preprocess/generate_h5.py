import h5py
import numpy as np
import cv2
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

negative_dir = './dataset/cropPicsNeg'
positive_dir = './dataset/cropPicsPos'

h5_save_path = './dataset/h5/data.h5'

SIZE = (96, 96)

def get_all_pics(dir_path):
    imgs = []
    # print('xxxxxx')
    for root, _, files in os.walk(dir_path):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            if(img is None):
                # print(os.path.join(root, file))
                continue
            img = cv2.resize(img, SIZE)
            # cv2.imshow('xx', img)
            # cv2.waitKey()
            imgs.append(img)
    return np.array(imgs)

pos_imgs = get_all_pics(positive_dir)
neg_imgs = get_all_pics(negative_dir)

print(pos_imgs.shape, neg_imgs.shape)

f = h5py.File(h5_save_path, 'w')
f['positive'] = pos_imgs
f['negative'] = neg_imgs
f.close()

f = h5py.File(h5_save_path, 'r')   
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
f.close()
