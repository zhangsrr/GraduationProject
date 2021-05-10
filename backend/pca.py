# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:04:26 2016
PCA source code
@author: liudiwei
"""

import numpy as np

#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征    
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示按照列来求均值，返回一维array，值为原array每列的均值，如果输入list,则axis=1


#计算方差,传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
def variance(X):
    m, n = np.shape(X)
    mu = meanX(X)
    muAll = np.tile(mu, (m, 1))    
    X1 = X - muAll
    variance = 1./m * np.diag(X1.T * X1)
    return variance

#标准化,传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
def normalize(X):
    m, n = np.shape(X)
    mu = meanX(X)
    muAll = np.tile(mu, (m, 1))    
    X1 = X - muAll
    X2 = np.tile(np.diag(X.T * X), (m, 1))
    XNorm = X1/X2
    return XNorm

"""
参数：
	- XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
	- k：表示取前k个特征值对应的特征向量
返回值：
	- finalData：参数一指的是返回的低维矩阵，对应于输入参数二
	- reconData：参数二对应的是移动坐标轴后的矩阵
"""


def pca(XMat, k):
    average = meanX(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))  # avgs是m行，每行一个average
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    # 如何避免inf和nan
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData, reconData