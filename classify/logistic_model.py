import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from preprocess.data_loader import data_loader
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from skimage.feature import hog
import pickle
from sklearn.utils import shuffle
import cv2
import time


class LogisticModel(object):
    def __init__(self):
        pass
    
    def load_data(self, x_training, y_training, x_testing, y_testing):
        self.x_training = np.concatenate((np.ones((x_training.shape[0], 1)), x_training), 1)
        self.x_testing = np.concatenate((np.ones((x_testing.shape[0], 1)), x_testing), 1)

        self.y_training = y_training
        self.y_testing = y_testing

        n_features = self.x_training.shape[1]
        self.n_samples = self.x_training.shape[0]

        self.weight = np.ndarray((n_features, 1))
    
    def train(self, batch_size=20, episode=20, lr=0.03, solver='sgd'):
        self.acc_list = {'training':[], 'testing':[]}
        for i_episode in range(episode):
            X, Y = shuffle(self.x_training, self.y_training)
            for i_batch in range(self.n_samples // batch_size):
                x = X[i_batch*batch_size: (i_batch+1)*batch_size, :]
                y = Y[i_batch*batch_size: (i_batch+1)*batch_size]
                self.update(x, y, lr, solver)
            
            acc_testing = self.evaluate(TrainingSet=False)
            acc_training = self.evaluate(TrainingSet=True)
            self.acc_list['testing'].append(acc_testing)
            self.acc_list['training'].append(acc_training)
            print('episode={0}, acc in training set={1}, acc in testing set = {2}'.format(i_episode, acc_training, acc_testing))
        self.plot()
        # self.save()
    
    def evaluate(self, TrainingSet):
        if(TrainingSet):
            X, y = self.x_training, self.y_training
        else:
            X, y = self.x_testing, self.y_testing
        y_predict = 1 / (1 + np.exp(-np.dot(X, self.weight)))
        y_predict[y_predict>=0.5] = 1
        y_predict[y_predict<0.5] = 0
        y_predict = np.squeeze(y_predict)
        wrong = np.count_nonzero(y.ravel() - y_predict)
        return (1 - float(wrong) / y.shape[0])
    
    def update(self, x, y, lr, solver='sgd'):
        grad = self.calculate_grad(x, y)
        if(solver == 'sgd'):
            self.weight += lr * grad
        elif(solver == 'langevin'):
            # print('grad')
            # print(0.5 * lr * grad)
            # print('second')
            # print(math.sqrt(lr) * np.random.normal(0, 1, 1))
            self.weight += lr * grad + math.sqrt(2*lr/200000) * np.random.normal(0, 1, 1)
        else:
            print('Fail to find the solver!')
    
    def calculate_grad(self, x, y):
        mul = np.dot(x, self.weight)
        y = np.resize(y, (y.shape[0], 1))
        grad = np.dot(np.transpose(x), y - np.exp(mul)/(1+np.exp(mul)))
        return grad

    def plot(self):
        if(len(self.acc_list['training']) > 0):
            episode = list(range(len(self.acc_list['training'])))
            plt.plot(episode, self.acc_list['training'], label='training')
        if(len(self.acc_list['testing']) > 0):
            episode = list(range(len(self.acc_list['testing'])))
            plt.plot(episode, self.acc_list['testing'], label='testing')
        plt.xlabel('episodes')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    
    def predict(self, imgs_list):
        features = []
        start = time.time()
        for img in imgs_list:
            img = cv2.resize(img, (96, 96))
            img = self.opencv2skimage(img)
            hog_feature = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
            hog_feature = np.insert(hog_feature, 0, 1) # add the bias 1 to the feature
            features.append(hog_feature)
        print('time for extracting hog : {}'.format(time.time() - start))
        start = time.time()
        features = np.array(features)
        y_predict = 1 / (1 + np.exp(-np.dot(features, self.weight)))
        print('time for prediction = {}'.format(time.time() - start))
        return y_predict.T[0]
    
    def opencv2skimage(self, src):
        cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = src.astype(np.float32)
        src /= 255
        return src
    
    def save(self, model_save_path):
        with open(model_save_path, 'wb') as f:
            pickle.dump(self.weight, f)
    
    def load(self, model_load_path):
        if(os.path.exists(model_load_path)):
            with open(model_load_path, 'rb') as f:
                self.weight = pickle.load(f)
        else:
            print('Fail to find the pretrainde model at {}!'.format(model_load_path))
