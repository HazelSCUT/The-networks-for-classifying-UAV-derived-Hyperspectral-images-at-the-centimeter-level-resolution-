# -*- coding: utf-8 -*-
"""
@author: Sonic
"""
#该代码是控制影像的输入和输出的，比如输入前将其标准化降维，输出时将它调色、评价精度等等

import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt
#对输入影像数据标准化
def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    #type==3 x = (2x-max(x))/(max(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
    elif type==3:
        maxX = np.max(X,0)
        X_norm = 2*X-maxX
        X_norm = X_norm/maxX
    return X_norm    
  #通过PCA实现降维
def PCANorm(X,num_PC):
    mu = np.mean(X,0)
    X_norm = X-mu
    
    Sigma = np.cov(X_norm.T)
    [U, S, V] = np.linalg.svd(Sigma)   
    XPCANorm = np.dot(X_norm,U[:,0:num_PC])
    return XPCANorm
#输出分类后的影像，palette是输出分类图像的调试版
def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:SCUT    
    #ID=7:Houston
    global palette
    global row
    global col
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 1145
        col = 1407
        palette = np.array([[0,128,0],
                            [50,205,50],
                            [0,255,0],
                            [192,192,192],
                            [245,245,245],
                            [47,79,79],
                            [255,255,255],
                            [0,0,0],
                            [178,34,34],
                            [30,144,255],
                            [128,128,0],
                            [255,248,220],
                            [175,238,238],
                            [0,0,255],
                            [189,183,107],
                            [105, 105, 105],
                            [240,128,128]])
        palette = palette*1.0/255    
    elif imageID == 7:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette*1.0/255
    
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
   #评价所选样本测试精度，即计算Kappa和OA
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA