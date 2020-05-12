# -*-coding:utf-8-*-
from __future__ import division, print_function, absolute_import
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
import scipy.io as sio
import h5py
import math
import os
from keras.callbacks import ModelCheckpoint
from net.DCAE_v2 import DCAE_v2_feature as DCAE_fea
from net.DCAE_v2 import DCAE_v2 as DCAE
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import time



def mkdir_if_not_exist(the_dir):
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)


def get_predict(model, x_data, shape, batch_size=32):
    shape = list(shape)
    nums = x_data.shape[0]
    shape[0] = nums
    x_predict = np.zeros(shape)
    for i in np.arange(int(nums / batch_size)):
        x_start, x_end = i*batch_size, (i+1)*batch_size
        data_temp = x_data[x_start:x_end, :, :, :, :]
        x_predict[x_start:x_end, :, :, :, :] = model.predict(data_temp)

    if nums % batch_size:
        x_start, x_end = nums - nums % batch_size, nums
        data_temp = x_data[x_start:x_end, :, :, :, :]
        x_predict[x_start:x_end, :, :, :, :] = model.predict(data_temp)
    return x_predict


def pre_process_data(name,patch_size):
    # 读取预处理的 .h5数据
    h5file_name = os.path.expanduser(
        './hyperspectral_datas/{}_patch_{}.h5'.format(name,patch_size))
    file = h5py.File(h5file_name, 'r')
    data = file['data'].value
    labels = file['labels'].value.flatten()
    return data, labels


def train_3DCAE_v3(x_train,x_test, model_name, n_epoch,data_name, patch_size = 5):
    model = DCAE(weight_decay=0.005)
    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['mse'])

    checkpointer = ModelCheckpoint(filepath=os.path.join(model_name,'epoch_{epoch:}.h5'), monitor='val_loss',verbose=1,
                            save_best_only=True, save_weights_only=False, period=1, mode='min',)

    history = model.fit(x_train, x_train, shuffle=True,nb_epoch = n_epoch,
                  validation_data=(x_test, x_test),
                  verbose=1, # verbose = 0 为不在标准输出流输出日志信息，verbose = 1 为输出进度条记录，verbose = 2 为每个epoch输出一行记录
                  callbacks=[checkpointer],
                  batch_size=60)
    print(history.history.keys())

    # summarize history for cosine
    plt.plot(history.history['mse'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['mse'], loc='upper left')
    plt.savefig('./model/trained_by_{}/{}_cosine.png'.format(data_name,data_name), dpi=600)
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./model/trained_by_{}/{}_loss.png'.format(data_name,data_name), dpi=600)
    plt.show()


def train_v3(model_name,n_epoch,data_name,patch_size = 5):

    x_train, _ = pre_process_data(data_name,patch_size=patch_size)
    X_train, X_test = train_test_split(x_train, train_size=0.95)
    train_3DCAE_v3(X_train, X_test, model_name=model_name,n_epoch=n_epoch,data_name = data_name)


class ModeTest:
    def __init__(self, model_name, epoch=2, data_name = 'data_name',data_row = None,data_col =None,patch_size=5):
        self.data, self.label = self.pre_process_data()
        self.model_name = model_name+'epoch_{}.h5'.format(epoch)
        self.data_name = data_name
        self.data_row = data_row
        self.data_col = data_col
        self.epoch = epoch
        self.patch_size = patch_size
        self.feature = None

    def pre_process_data(self):
        h5file_name = os.path.expanduser(
            './hyperspectral_datas/{}_patch_{}.h5'.format(data_name,patch_size))
        file = h5py.File(h5file_name, 'r')
        data = file['data'].value
        labels = file['labels'].value.flatten()
        return data, labels

    def get_latent_feature(self):
        feature_model = DCAE_fea()
        feature_model.load_weights(self.model_name, by_name=True)
        self.feature = get_predict(
            model=feature_model, x_data=self.data, shape=feature_model.predict(self.data[:2]).shape)
        # 融合潜在特征
        self.feature = np.mean(self.feature, 4, keepdims=True)
        print("latent image shape:",self.feature.shape)
        plt.imshow(self.feature[:, 1, 0, 0, 0].reshape((self.data_row, self.data_col)))
        plt.show()


    def save_feature_label(self,save_file_name):
        file = h5py.File(save_file_name, 'w')
        file.create_dataset('feature', data=self.feature)
        file.create_dataset('label', data=self.label)
        file.close()

if __name__ == '__main__':

    optional = 'train'
    # 所要训练数据的名称
    data_name = 'indian_pines_corrected'
    patch_size = 5
    model_name = './model/trained_by_{}/'.format(data_name)
    # 训练
    if optional == 'train':
        train_v3(model_name, n_epoch=10, data_name=data_name, patch_size=patch_size)
    # 测试
    elif optional == 'test':
        PCA_data = sio.loadmat('./hyperspectral_datas/PCA_{}.mat'.format(data_name))['data']
        row, col, band = PCA_data.shape
        test_mode = ModeTest(model_name=model_name,epoch=10, data_name=data_name,
                             data_row=row, data_col=col, patch_size=patch_size)
        # 获取潜在特征层的数据
        test_mode.get_latent_feature()
        location1 = './result/{}/{}_CAE_latent_feature.h5'.format(data_name, data_name)
        test_mode.save_feature_label(location1)
        data1 = h5py.File(location1, 'r')
        data1 = data1['feature'].value
        n_hidden = len(data1[1])
        # print('latent image dim:',n_hidden)
        data1 = data1[:, :, 0, 0, 0].reshape(row, col, n_hidden) # n_hidden是潜在特征维度
        sio.savemat('result/{}/{}_latent_image.mat'.format(data_name, data_name), {'data': data1})






