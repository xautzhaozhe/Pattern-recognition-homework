# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
from sklearn import svm
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


Label = sio.loadmat('./hyperspectral_datas/Indian_pines_gt.mat')['indian_pines_gt']
Data = sio.loadmat('./result/indian_pines_corrected/indian_pines_corrected_latent_image.mat')['data']

imGIS = Label
im = Data

Data = np.reshape(Data, (Data.shape[0]*Data.shape[1],Data.shape[2]))
Label = np.reshape(Label, (Label.shape[0]*Label.shape[1]))
Data = Data[Label > 0, :]
Label = Label[Label > 0]

X_train, X_test, y_train, y_test = train_test_split(Data, Label, train_size=0.3,
                                                    random_state=345, stratify=Label)

test_bg = True
# 用SVM进行训练,参数C在误分类样本和分界面简单性之间进行权衡；
# gamma表示单个样本对训练的影响，值越大影响越小；
clf = svm.SVC(C=100, kernel='linear', gamma=1)
# ’linear’，‘poly’，‘rbf’，‘sigmoid’，‘precomputed’或者callable之一
train = np.asarray(X_train)
train_label = np.asarray(y_train)
clf.fit(train, train_label)


C = np.max(imGIS)
matrix = np.zeros((C,C))
for i in range(len(X_test)):
    r = clf.predict(X_test[i].reshape(-1,len(X_test[i])))
    matrix[r-1, y_test[i]-1] += 1

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
sio.savemat(os.path.join('SVM','result', 'result.mat'), {'oa': accuracy,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})

iG = np.zeros((imGIS.shape[0], imGIS.shape[1]))
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
plt.savefig(os.path.join('SVM','result', 'decode_map'+str(int(test_bg))+'.png'), format='png')
plt.close()
print('decode map get finished')


