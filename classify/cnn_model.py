import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

from preprocess.data_loader import data_loader
import torch
from torch import nn
from torch.autograd import  Variable
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import cv2
from skimage.feature import hog
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #input: 96*96*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=3,
                padding=1
            ), #32*32*16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4) 
        )
        #8*8*16
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            #4*4*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #2*2*32
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=1,
                padding=0,
            ), #2*2*32
            nn.ReLU()
        )
        #1*1*32
        self.linear = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.linear(x)

        return output


class CNNModel(object):
    def __init__(self):
        pass
    
    def load_data(self, x_training, y_training, x_testing, y_testing):
        self.x_training, self.y_training = x_training, y_training
        self.x_testing, self.y_testing = x_testing, y_testing
            
    def train(self, max_epoch=10, batch_size=32):
        self.cnn = CNN().cuda()

        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=0.001)

        self.losses_train = []
        self.losses_test = []
        self.accs_test = []
        for epoch in range(max_epoch):
            # break
            X, y = shuffle(self.x_training, self.y_training)
            loss_training = 0
            for i_batch in range(len(X)//batch_size):
                # print(X.shape, y.shape)
                x_input = Variable(torch.from_numpy(X[i_batch*batch_size:(i_batch+1)*batch_size])).float().permute(0, 3, 1, 2)
                x_input = x_input.cuda()
                y_predict = self.cnn.forward(x_input).squeeze()
                y_real = Variable(torch.from_numpy(y[i_batch*batch_size:(i_batch+1)*batch_size]).squeeze()).float()
                y_real = y_real.cuda()
                loss = nn.BCELoss().cuda()(y_predict, y_real)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_training += loss

            loss_training /= len(X)//batch_size
            self.losses_train.append(loss_training)

            # test
            if epoch % 1 == 0:
                cnn_test = CNN()
                cnn_test.load_state_dict(self.cnn.state_dict())
                x_input = Variable(torch.from_numpy(self.x_testing)).float().permute(0, 3, 1, 2)
                y_predict = cnn_test.forward(x_input).squeeze()
                y_real = Variable(torch.from_numpy(self.y_testing).squeeze()).float()
                with torch.no_grad():
                    loss_testing = nn.BCELoss()(y_predict, y_real)
                y_predict[y_predict > 0.5] = 1
                y_predict[y_predict <= 0.5] = 0
                y_predict = y_predict.long()
                y_real = y_real.long()
                acc = ((y_predict == y_real).sum().float() / float(y_real.shape[0]))
                print('epoch {}: loss training={}, loss testing={}, acc testing={}'.format(epoch, loss_training, loss_testing, acc))
                self.accs_test.append(acc)
                self.losses_test.append(loss_testing)
        self.plot()
        # self.save()
    
    def plot(self):
        X = list(range(len(self.accs_test)))
        plt.subplot(2,1,1)
        plt.title('Accuracy in the testing set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(X, self.accs_test)

        plt.subplot(2,1,2)
        plt.title('Loss long with epoches')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(X, self.losses_test, label='Testing set')
        plt.plot(X, self.losses_train, label='Training set')

        plt.legend()
        plt.show()

    def save(self, model_save_path):
        torch.save(self.cnn.state_dict(), model_save_path)
    
    def load(self, model_load_path):
        self.cnn = CNN().cuda()
        if(os.path.exists(model_load_path)):
            self.cnn.load_state_dict(torch.load(model_load_path))
        else:
            print('Fail to find the pretrainde model at {}!'.format(model_load_path))
    
    def predict(self, imgs_list, cuda=True):
        batch_size = 64
        p = None
        if not cuda:
            cnn_test = CNN()
            cnn_test.load_state_dict(self.cnn().state_dict())
        for i_batch in range(0, len(imgs_list)//batch_size+1):
            if (i_batch + 1) * batch_size > len(imgs_list):
                input_len =  -i_batch * batch_size + len(imgs_list)
            else:
                input_len = batch_size
            x_input = np.ndarray((input_len, 96, 96, 3))
            for i_img in range(input_len):
                img = imgs_list[i_batch*batch_size + i_img]
                img = cv2.resize(img, (96, 96))
                x_input[i_img, ...] = img
            x_input = torch.from_numpy(x_input).float().permute(0, 3, 1, 2)
            if cuda:
                x_input = x_input.cuda()
                with torch.no_grad():
                    p_batch = self.cnn(x_input).squeeze()
            else:
                with torch.no_grad():
                    p_batch = cnn_test(x_input).squeeze()
            
            p_batch = p_batch.cpu().detach().numpy()
            if p is None:
                p = p_batch
            else:
                p = np.append(p, p_batch)
        return p

