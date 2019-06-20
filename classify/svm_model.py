import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from preprocess.data_loader import data_loader
import shutil
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
import os
import pickle
import cv2
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class SVMModel(object):
    def __init__(self):
        pass
    
    def load_data(self, x_training, y_training, x_testing, y_testing):
        self.x_training, self.y_training, self.index_training = shuffle(x_training, y_training, np.arange(0, x_training.shape[0]))
        self.x_testing, self.y_testing = x_testing, y_testing
    
    def train(self, kernel):
        self.clf = svm.SVC(kernel=kernel, probability=True)

        print('start training ...')
        self.clf.fit(self.x_training, self.y_training.ravel())
        sv_neg, sv_pos = self.clf.n_support_[0], self.clf.n_support_[1]
        sv_total = sv_neg + sv_pos
        print('total support vectors number={}, negative sv ={}, positive sv={}'.format(sv_total, sv_neg, sv_pos))
        self.support_vectors_label = self.y_training[self.clf.support_].astype(int)

        acc_test = self.clf.score(self.x_testing, self.y_testing)
        acc_train = self.clf.score(self.x_training, self.y_training)
        print('acc training set = {}, acc testing set = {}'.format(acc_train, acc_test))

        # self.save()

    def save(self, model_save_path):
        with open(model_save_path, 'wb') as f:
            pickle.dump({'clf':self.clf, 'label':self.support_vectors_label}, f)
    
    def load(self, model_load_path):
        if(os.path.exists(model_load_path)):
            with open(model_load_path, 'rb') as f:
                m = pickle.load(f)
                self.clf = m['clf']
                self.support_vectors_label = m['label']
        else:
            print('Fail to find the pretrainde model at {}!'.format(model_load_path))

    def predict(self, imgs_list):
        features = []
        for img in imgs_list:
            img = cv2.resize(img, (96, 96))
            img = self.opencv2skimage(img)
            hog_feature = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
            features.append(hog_feature)
        features = np.array(features)
        p = self.clf.predict(features)
        return p
        # p = self.clf.predict(hog_feature)
        
    
    def opencv2skimage(self, src):
        cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = src.astype(np.float32)
        src /= 255
        return src

    def show_support_vectors(self):
        label = self.support_vectors_label.ravel()
        tsne = TSNE(2, perplexity=50, early_exaggeration=100.0, n_iter=100000)
        X_tsne = tsne.fit_transform(self.clf.support_vectors_)

        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)

        sv_neg, sv_pos = self.clf.n_support_[0], self.clf.n_support_[1]
        sv_total = sv_neg + sv_pos
        print('total support vectors number={}, negative sv ={}, positive sv={}'.format(sv_total, sv_neg, sv_pos))
        print(X_norm[:, 0][label==1].shape, X_norm[:, 0][label==0].shape)

        plt.scatter(X_norm[:, 0][label==1], X_norm[:, 1][label==1], color='r')
        plt.scatter(X_norm[:, 0][label==0], X_norm[:, 1][label==0], color='g')
        plt.show()

    def find_support_vectors(self):
        self.show_support_vectors()
        sv_save_path_pos = './classify/SupportVectors/sv_pos.txt'
        sv_save_path_neg = './classify/SupportVectors/sv_neg.txt'

        img_root_path_pos = './dataset/cropPicsPos'
        hog_index_save_pos = './dataset/hog/hog_pos_index.txt'

        img_root_path_neg = './dataset/cropPicsNeg'
        hog_index_save_neg = './dataset/hog/hog_neg_index.txt'

        with open(hog_index_save_pos, 'r') as f:
            list_pos = [x.strip() for x in f.readlines()]
        with open(hog_index_save_neg, 'r') as f:
            list_neg = [x.strip() for x in f.readlines()]

        img_list_neg = []
        img_list_pos = []
        for i in self.clf.support_:
            if(self.y_training[i] == 0):
                j = self.index_training[i]
                img_list_neg.append(list_neg[j])
            elif(self.y_training[i] == 1):
                j = self.index_training[i]
                img_list_pos.append(list_pos[j])

        with open(sv_save_path_pos, 'w') as f:
            for img_name in img_list_pos:
                f.write(img_name + '\n')
        with open(sv_save_path_neg, 'w') as f:
            for img_name in img_list_neg:
                f.write(img_name + '\n')
        
        for img_name in img_list_pos[:10]:
            img = cv2.imread(os.path.join(img_root_path_pos, img_name + '.jpg'))
            cv2.imshow('positive support vectors', img)
            cv2.waitKey(1000)
        
        for img_name in img_list_neg[:10]:
            img = cv2.imread(os.path.join(img_root_path_neg, img_name + '.jpg'))
            cv2.imshow('negative support vectors', img)
            cv2.waitKey(1000)
        