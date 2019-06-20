import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from skimage.feature import hog
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np

img_root_path_pos = './dataset/cropPicsPos'
list_dir_pos = './dataset/cropPicsPos/list'
hog_img_save_pos ='./dataset/hog/hog_pos.npy'
hog_index_save_pos = './dataset/hog/hog_pos_index.txt'

img_root_path_neg = './dataset/cropPicsNeg'
list_dir_neg = './dataset/cropPicsNeg/list'
hog_img_save_neg ='./dataset/hog/hog_neg.npy'
hog_index_save_neg = './dataset/hog/hog_neg_index.txt'

def general_hog(img_root_path, list_dir, hog_save_path, hog_index_path):
    list_txt_list = os.listdir(list_dir)
    hog_features = []
    total_name_list = []
    for list_txt in list_txt_list:
        f_list = open(os.path.join(list_dir, list_txt))
        while(True):
            img_name = f_list.readline().strip()
            print(img_name)
            if(img_name == ''):
                break
            total_name_list.append(img_name)
            img_path = img_name + '.jpg'
            img_ori = io.imread(os.path.join(img_root_path, img_path), as_gray=False)
            img_ori = resize(img_ori, (96, 96))
            hog_feature = hog(img_ori, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
            hog_features.append(hog_feature)

            # print(type(hog_feature))
            # print(hog_feature.shape)

            # full_path = os.path.join(hog_img_save_dir, img_path)
            # dir_path, _ = os.path.split(full_path)
            # if(not os.path.exists(dir_path)):
            #     os.makedirs(dir_path)
            # io.imsave(full_path, hog_image_ori)

            # plt.subplot(1, 2, 1)
            # io.imshow(img_ori)
            # plt.subplot(1, 2, 2)
            # io.imshow(hog_image_ori)
            # io.show()

        f_list.close()
    hog_features = np.array(hog_features)
    np.save(hog_save_path, hog_features)

    with open(hog_index_path, 'w') as f:
        for img_name in total_name_list:
            f.write(img_name+'\n')


if __name__ == '__main__':
    general_hog(img_root_path_pos, list_dir_pos, hog_img_save_pos, hog_index_save_pos)
    general_hog(img_root_path_neg, list_dir_neg, hog_img_save_neg, hog_index_save_neg)