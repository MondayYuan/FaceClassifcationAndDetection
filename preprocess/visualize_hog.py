from skimage.feature import hog
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

img_root_folder = './dataset/originalPics'
list_txt_path = './dataset/FDDB-folds/FDDB-fold-01.txt'


f_list =  open(list_txt_path, 'r')
while(True):
    img_name = f_list.readline().strip()
    print(img_name)
    img_path = img_name + '.jpg'
    img_ori = io.imread(os.path.join(img_root_folder, img_path), as_gray=False)
    img_ori = resize(img_ori, (96, 96))
    if(img_name == ''):
        break
    _, hog_image_ori = hog(img_ori, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    plt.subplot(1, 2, 1)
    io.imshow(img_ori)
    plt.subplot(1, 2, 2)
    io.imshow(hog_image_ori)
    io.show()

f_list.close()

