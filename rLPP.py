# -*- coding: utf-8 -*-
# Author:       Liangliang
# Date:         2019\3\30 0030 20:04:04
# File:         rLPP.py
# Software:     PyCharm
#------------------------------------

import numpy as np
import math

def f(X,S,W,p):#计算目标优化函数值
     f = 0 #初始化目标函数值
     for i in range(X.shape[1]):
         for j in range(X.shape[1]):
             f = f + S[i,j]*math.pow(np.linalg.norm(np.dot(W.transpose(),X[:,i].reshape(X.shape[0],1))-np.dot(W.transpose(),X[:,j].reshape(X.shape[0],1)), ord=2),p)
     return f;


def rLPP(data,d):
    '''
    此算法执行的是rLPP算法,主要的功能是实现数据的降维
	Wang H, Nie F, Huang H. Learning robust locality preserving projection via p-order minimization[C]//Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015:3059-3065.
    data: 数据n*m,每一行代表一个样本,每一列代表一个特征 d:降维后数据的维度
    return: 返回的是一个n*d的数据,数据每一行代表一个样本,每一列代表一个特征
    '''
    N_MAX = 100#最大迭代次数设置为100
    X = data.transpose() #将数据转化成列形式,与文中的公式保持一致 m*n
    n = data.shape[0] #数据的样本数
    m = data.shape[1] #数据的维数
    #初始化参数
    W = np.random.rand(m,d) #初始化矩阵
    p = 0.3 #代表文中计算的是矩阵的p阶范数
    #计算样本间的相似度,可以认为是构建图的邻接矩阵
    S = np.zeros((data.shape[0], data.shape[0])) #相似度矩阵
    D = np.zeros((data.shape[0], data.shape[0])) #度矩阵
    for i in range(n):
        for j in range(n):
            S[i,j] = math.exp(-np.linalg.norm(data[i,:] - data[j,:], ord=2))#计算相似度矩阵
        D[i,i] = sum(S[i,:])#计算度矩阵
    #初始化迭代过程中的变量
    f_best = np.inf #目标函数值的最优值
    W_befor = W;
    for num in range(N_MAX):#进行迭代
        if f(X,S,W,p) < f_best:#迭代后目标函数值减小
            #计算新的矩阵
            St = np.zeros((data.shape[0], data.shape[0]))  # 初始化(4)式定义的相似度矩阵
            Dt = np.zeros((data.shape[0], data.shape[0]))  # 初始化(4)式定义的度矩阵
            for i in range(n):
                for j in range(n):
                    St[i, j] = S[i, j] * math.pow(np.linalg.norm(
                        np.dot(W.transpose(), X[:, i].reshape(X.shape[0], 1)) - np.dot(W.transpose(),X[:, j].reshape(X.shape[0], 1)),ord=2), p)  # 更新相似度矩阵值
                Dt[i, i] = sum(St[i, :])  # 更新度矩阵值
            Lt = Dt - St #计算新的拉普拉斯矩阵
            vector_values, vectors = np.linalg.eig(np.dot(np.linalg.inv(np.dot(np.dot(X,D),X.transpose())), np.dot(X, np.dot(Lt,X.transpose()))))#求解特征值与特征向量
            index = np.argsort(vector_values)#对特征值进行有小到大排序
            W = np.zeros((m,d))
            for k in range(1,d+1):#选取排序后第2个至第d+1个特征值对应的特征向量
                W[:,k-1] = vectors[:,k]
            if f(X,S,W,p) < f_best:#新的参数继续减小,进行下一次迭代
                f_best = f(X,S,W,p)
                W_before = W
            else:#目标函数值不再减小,算法终止返回当前的W
                W = W_before
                break
        else:
            W = W_before
            break
    X = np.dot(W.transpose(),X)#获得最终的结果
    X = X.transpose()#把X转换成与输入一致的形式,每一行代表一个样本,每一个列代表一个特征
    return X
