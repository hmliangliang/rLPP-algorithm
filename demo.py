# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\3\30 0030 22:08:08
# File:         demo.py
# Software:     PyCharm
#------------------------------------
import numpy as np
from scipy.io import loadmat
import time
import rLPP
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    d=8
    data = loadmat('Breastw.mat')
    data = data['Breastw']
    data = np.array(data)#转换成numpy数组形式
    label = data[:,data.shape[1]-1].reshape(data.shape[0],1)#获取类标签
    data = data[:,0:data.shape[0]-1]#获取数据部分
    d = min(d,data.shape[1])#防止d设置超过了原先数据的维度
    data = rLPP.rLPP(data,d)
    num = int(2*data.shape[0]/3)
    model = KNeighborsClassifier(n_neighbors = 5)#训练分类器
    model.fit(data[0:num,:],label[0:num,0])
    y = model.predict(data[num:data.shape[0],:])#预测结果
    print('预测结果的准确率为:',sum(y==label[num:data.shape[0],0])/(data.shape[0]-num))
