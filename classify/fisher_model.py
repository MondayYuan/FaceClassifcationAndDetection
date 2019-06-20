import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from preprocess.data_loader import data_loader
import pickle
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

class FisherModel(object):
    def __init__(self):
        pass

    def load_data(self, x_training, y_training, x_testing, y_testing):
        self.x_training, self.y_training = x_training, y_training.ravel()
        self.x_testing, self.y_testing = x_testing, y_testing
        self.pos_samples, self.neg_samples = self.x_training[self.y_training==1], self.x_training[self.y_training==0]

    def train(self):
        self.fit(self.pos_samples, self.neg_samples)
        acc_testing = self.evalute(self.x_testing, self.y_testing)
        acc_training = self.evalute(self.x_training, self.y_training)
        print('acc in training set = {}, acc in testing set = {}'.format(acc_training, acc_testing))
        # self.save()

    def fit(self, pos_samples, neg_samples):
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.n_samples_pos = self.pos_samples.shape[0]
        self.n_samples_neg = self.neg_samples.shape[0]
        self.n_features = self.pos_samples.shape[1]

        self.pos_mean = np.resize(np.mean(self.pos_samples, 0), (1, self.n_features))
        # print(self.pos_mean)
        self.neg_mean = np.resize(np.mean(self.neg_samples, 0), (1, self.n_features))

        s1 = np.dot(np.transpose(self.pos_samples - self.pos_mean), self.pos_samples - self.pos_mean)
        s2 = np.dot(np.transpose(self.neg_samples - self.neg_mean), self.neg_samples - self.neg_mean)
        self.sw = s1 + s2
        
        self.weight = np.dot(np.linalg.inv(self.sw), np.transpose(self.neg_mean - self.pos_mean))

        self.y_threshold = (np.dot(self.pos_mean, self.weight) + np.dot(self.neg_mean, self.weight)) / 2
        self.y_threshold = self.y_threshold[0][0]

        self.intra_class_variance = np.dot(np.dot(self.weight.T, self.sw), self.weight)[0][0]
        self.inter_class_variance = (np.dot(self.pos_mean - self.neg_mean, self.weight) ** 2)[0][0]

        print('intra-class-variance={}, inter-class-variance={}'.format(self.intra_class_variance, self.inter_class_variance))
        self.plot()
    
    def plot(self):
        y_pos = np.dot(self.pos_samples, self.weight)
        y_neg = np.dot(self.neg_samples, self.weight)

        y_pos = y_pos[:100]
        y_neg = y_neg[:100]

        plt.scatter(np.arange(0, len(y_pos)), y_pos, color='r', label='postive')
        plt.scatter(np.arange(0, len(y_neg)), y_neg, color='g', label='negative')
        plt.plot([0, len(y_neg)], [self.y_threshold, self.y_threshold])

        plt.legend()
        plt.show()

        # print(self.intra_class_variance, self.inter_class_variance)
    
    def evalute(self, X, y):
        y_predict = np.dot(X, self.weight)
        
        y_predict[y_predict>self.y_threshold] = 0
        y_predict[y_predict<=self.y_threshold] = 1
        y_predict = np.squeeze(y_predict)
        
        wrong = np.count_nonzero(y - y_predict)
        right = y.shape[0] - wrong
        acc = float(right) / (wrong + right)
        return acc
    
    def save(self, model_save_path):
        info = {
            'weight': self.weight,
            'threshold': self.y_threshold,
            'intra_class_variance': self.intra_class_variance,
            'inter_class_variance': self.inter_class_variance
        }

        with open(model_save_path, 'wb') as f:
            pickle.dump(info, f)
    
    def load(self, model_load_path):
        if(os.path.exists(model_load_path)):
            with open(model_load_path, 'rb') as f:
                info = pickle.load(f)
                self.weight = info['weight']
                self.y_threshold = info['threshold']
                self.inter_class_variance = info['inter_class_variance']
                self.intra_class_variance = info['intra_class_variance']
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
        y_predict = np.dot(features, self.weight)

        return float(y_predict[0] <= self.y_threshold)

    def opencv2skimage(self, src):
        cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = src.astype(np.float32)
        src /= 255
        return src
    
