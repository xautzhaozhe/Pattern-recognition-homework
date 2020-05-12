# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import os

path = os.path.join("./hyperspectral_datas/")# change this path for your dataset
PaviaU = os.path.join(path,'PaviaU.mat')
PaviaU_gt = os.path.join(path,'PaviaU_gt.mat')
method_path = '3DCAE_SVM'

# 加载数据
data = sio.loadmat(PaviaU)
data_gt = sio.loadmat(PaviaU_gt)
im = sio.loadmat('./result/latent')
print('dataset shape:',im.shape)
show_data = im[:,:,25]
plt.imshow(show_data)
plt.show()
imGIS = data_gt['paviaU_gt']# gt
plt.imshow(imGIS)
plt.show()
# 归一化
max = np.max(im) - np.min(im)
im = (im - float(np.min(im)))
im = im/max

sample_num = 100
deepth = im.shape[2]
classes = np.max(imGIS)
test_bg = False

data_pos = {}
train_pos = {}
test_pos = {}

for i in range(1,classes+1):
    data_pos[i] = []
    train_pos[i] = []
    test_pos[i] = []

# 把某一类对应的索引，放入到data_pos[]里面
for i in range(imGIS.shape[0]):
    for j in range(imGIS.shape[1]):
        for k in range(1,classes+1):
            if imGIS[i,j]==k:
                data_pos[k].append([i,j])
                continue # 终止此次循环，而不是终止整个循环


# 从每一类样本中选择200个样本作为训练集；剩余的是测试集（获取训练样本和测试样本的索引）
for i in range(1,classes+1):
    # random.sample函数，从data_pos[]中随机选择sample_num个样本
    indexies = random.sample(range(len(data_pos[i])),sample_num)
    for k in range(len(data_pos[i])):
        if k not in indexies:
            test_pos[i].append(data_pos[i][k])
        else:
            train_pos[i].append(data_pos[i][k])


train = []
train_label = []
test = []
test_label = []

# 获取训练数据和训练标签
for i in range(1,len(train_pos)+1):
    for j in range(len(train_pos[i])):
        row,col = train_pos[i][j]
        train.append(im[row,col])
        train_label.append(i)
# 获取测试数据和标签
for i in range(1,len(test_pos)+1):
    for j in range(len(test_pos[i])):
        row,col = test_pos[i][j]
        test.append(im[row,col])
        test_label.append(i)
# 建立实验结果路径
if not os.path.exists(os.path.join(method_path,'result')):
    os.makedirs(os.path.join(method_path,'result'))
# 用SVM进行训练,参数C在误分类样本和分界面简单性之间进行权衡；
# gamma表示单个样本对训练的影响，值越大影响越小；
clf = SVC(C=100, kernel='rbf', gamma=1)
train = np.asarray(train)
train_label = np.asarray(train_label)
clf.fit(train,train_label)


C = np.max(imGIS)
matrix = np.zeros((C,C))
for i in range(len(test)):
    r = clf.predict(test[i].reshape(-1,len(test[i])))
    matrix[r-1,test_label[i]-1] += 1

ac_list = []
for i in range(len(matrix)):
    ac = matrix[i, i] / sum(matrix[:, i])
    ac_list.append(ac)
    print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)

# print('confusion matrix:')
# print(np.int_(matrix))
print('total right num:', np.sum(np.trace(matrix)))
print('total test num:',np.sum(matrix))
accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
print('Overall accuracy:', accuracy)
# kappa
kk = 0
for i in range(matrix.shape[0]):
    kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
pe = kk / (np.sum(matrix) * np.sum(matrix))
pa = np.trace(matrix) / np.sum(matrix)
kappa = (pa - pe) / (1 - pe)
ac_list = np.asarray(ac_list)
aa = np.mean(ac_list)
print('Average accuracy:',aa)
print('Kappa:', kappa)
sio.savemat(os.path.join(method_path,'result', 'result.mat'), {'oa': accuracy,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
iG = np.zeros((imGIS.shape[0],imGIS.shape[1]))
for i in range(imGIS.shape[0]):
    for j in range(imGIS.shape[1]):
        if imGIS[i,j] == 0:
            if test_bg:
                iG[i,j] = (clf.predict(im[i,j].reshape(-1,len(im[i,j]))))
            else:
                iG[i,j]=0
        else:
            iG[i,j] = (clf.predict(im[i,j].reshape(-1,len(im[i,j]))))
if test_bg:
    iG[0,0] = 0
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
plt.axis('off')
plt.pcolor(iG, cmap='jet')
plt.savefig(os.path.join(method_path,'result', 'decode_map'+str(int(test_bg))+'.png'), format='png')
plt.close()
print('decode map get finished')


