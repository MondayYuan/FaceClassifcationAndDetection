import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from skimage import io
import numpy as np
import h5py
from sklearn.utils import shuffle
import cv2
import sys

hog_pos_path = './dataset/hog/hog_pos.npy'
hog_neg_path = './dataset/hog/hog_neg.npy'

h5_path = './dataset/h5/data.h5'

def data_loader(from_h5=False, mode='train'):
    if from_h5:
        f = h5py.File(h5_path, 'r')
        all_pos_samples = f['positive']
        all_neg_samples = f['negative']
    
    else:
        all_pos_samples = np.load(hog_pos_path)
        all_neg_samples = np.load(hog_neg_path)

    if mode == 'train':
        ratio = 0.8
    elif mode == 'test':
        ratio = 0.2
    neg_num = int(ratio * all_neg_samples.shape[0])
    pos_num = int(ratio * all_pos_samples.shape[0])

    pos_label = np.ones((pos_num, 1))
    neg_label = np.zeros((neg_num, 1))

    if mode == 'train':
        X = np.concatenate((all_pos_samples[:pos_num], all_neg_samples[:neg_num]), axis=0)
    elif mode == 'test':
        X = np.concatenate((all_pos_samples[-pos_num:], all_neg_samples[-neg_num:]), axis=0)

    Y = np.concatenate((pos_label, neg_label), axis=0)

    print('{} : negative={}, positive={}, total={}'.format(mode, neg_num, pos_num, neg_num+pos_num))

    return X, Y.astype(np.int).ravel()