# -*- coding: utf-8 -*-
"""
@author: Sonic
"""

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral,create_pairwise_gaussian, softmax_to_unary
import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt
import time
import sys
import os
from HyperFunctions import *
import tensorflow as tf

resultpath = './SSFCN/'
if not os.path.isdir(resultpath):
    os.makedirs(resultpath)

def weight_variable(shape,name=None):    
    initial = tf.random_uniform(shape,minval=0.01, maxval=0.02)
    return tf.Variable(initial,name)

def weight(name=None):
    initial = tf.random_uniform([1],minval=1, maxval=2)
    return tf.Variable(initial,name)

def bias_variable(shape,name=None):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial,name=None)

def conv2d(x, W, p=0):
    if p==1:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    else:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        
def atrousconv2d(x,W,rate):
    return tf.nn.atrous_conv2d(x, W,rate,padding='SAME')

def max_pool(x,W,p=0):
    if p==1:
        return tf.nn.max_pool(x, ksize=[1, W, W, 1],
                        strides=[1, W, W, 1], padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, W, W, 1],
                        strides=[1, W, W, 1], padding='VALID')        
                        
def max_pool_padding(x,W):
    return tf.nn.max_pool(x, ksize=[1, W, W, 1],
                        strides=[1, 1, 1, 1], padding='SAME')
                                               
def avg_pool_padding(x,W):
    return tf.nn.avg_pool(x, ksize=[1, W, W, 1],
                        strides=[1, 1, 1, 1], padding='SAME')                        
                        
def one_hot3d(Y):
    num_class = np.max(Y)
    row,col = Y.shape
    y = np.zeros((row,col,num_class),'uint8')
    for i in range(1,num_class+1):
        index = np.where(Y==i)
        y[index[0],index[1],i-1] = 1
    return y         

x = tf.placeholder(tf.float32, shape=[1,1145,1407,83])
y_ = tf.placeholder(tf.float32, shape=[1,1145,1407,17])

idx1 = tf.placeholder(tf.int32, shape=None)
idx2 = tf.placeholder(tf.int32, shape=None)
idx3 = tf.placeholder(tf.int32, shape=None)

W_spectral_conv1 = weight_variable([1,1,83,128],'W_spectral_conv1')
b_spectral_conv1 = bias_variable([128],'b_spectral_conv1')
h_spectral_conv1 = tf.nn.relu(conv2d(x, W_spectral_conv1,1) + b_spectral_conv1)

W_spectral_conv2 = weight_variable([1,1, 128, 128],'W_spectral_conv2')
b_spectral_conv2 = bias_variable([128],'b_spectral_conv2')
h_spectral_conv2 = tf.nn.relu(conv2d(h_spectral_conv1, W_spectral_conv2,1) + b_spectral_conv2)

W_spectral_conv3 = weight_variable([1,1,128,128],'W_spectral_conv3')
b_spectral_conv3 = bias_variable([128],'b_spectral_conv3')
h_spectral_conv3 = tf.nn.relu(conv2d(h_spectral_conv2, W_spectral_conv3,1) + b_spectral_conv3)

h_spectral = h_spectral_conv1 + h_spectral_conv2 + h_spectral_conv3

rate=2

W_dr_conv1 = weight_variable([1,1, 83, 128],'W_dr_conv1')
b_dr_conv1 = bias_variable([128],'b_dr_conv1')
h_dr_conv1 = tf.nn.relu(conv2d(x, W_dr_conv1,1) + b_dr_conv1)

W_spatial_conv1 = weight_variable([5,5,128,128],'W_spatial_conv1')
W_spatial_conv1 = tf.nn.dropout(W_spatial_conv1,rate=0.1)  # set dropout rate as 0.1
b_spatial_conv1 = bias_variable([128],'b_spatial_conv1')
h_spatial_conv1 = tf.nn.relu(atrousconv2d(h_dr_conv1, W_spatial_conv1,2) + b_spatial_conv1)
h_spatial_pool1 = avg_pool_padding(h_spatial_conv1,2)

W_spatial_conv2 = weight_variable([5,5, 128, 128],'W_spatial_conv2')
W_spatial_conv2 = tf.nn.dropout(W_spatial_conv2,rate=0.1)  # set dropout rate as 0.1
b_spatial_conv2 = bias_variable([128],'b_spatial_conv2')
h_spatial_conv2 = tf.nn.relu(atrousconv2d(h_spatial_pool1, W_spatial_conv2,2) + b_spatial_conv2)
h_spatial_pool2 = avg_pool_padding(h_spatial_conv2,2)

h_spatial = h_dr_conv1 + h_spatial_pool1 + h_spatial_pool2

W_spatial = weight('W_spatial')
W_spectral = weight('W_spectral')

h_SS = W_spatial*h_spatial + W_spectral*h_spectral

W_conv5 = weight_variable([1,1, 128, 17],'W_conv5')
b_conv5 = bias_variable([17],'b_conv5')
y_conv = tf.nn.relu(conv2d(h_SS, W_conv5,1) + b_conv5)

y_prob = tf.nn.softmax(y_conv)
y_label = tf.argmax(y_prob,-1)

indices = tf.stack([idx3,idx1,idx2], axis=1)

y_conv_mask = tf.gather_nd(y_conv, indices)
y_mask = tf.gather_nd(y_, indices)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv_mask, labels=y_mask))
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv_mask,-1), tf.argmax(y_mask,-1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

X = sio.loadmat('./DataSets/SubBandHSI.mat')['SubBandHSI'].astype('float32')
Y = sio.loadmat('./DataSets/sample17.mat')['sample17']

row,col,n_band = X.shape
X = np.reshape(featureNormalize(np.reshape(X,-1),2),(row,col,n_band))

num_class = np.max(Y)
Y_train = np.zeros(Y.shape).astype('int')
n_sample_train = 0
n_sample_test = 0

FCN_joint = np.zeros((num_class+3))
FCN_crf = np.zeros((num_class+3))

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   
init = tf.global_variables_initializer()  
sess.run(init)

for i in range(1,num_class+1):
    index = np.where(Y==i)
    n_sample = len(index[0])
    array = np.random.permutation(n_sample)
    n_per = 150
    if i==1:
        array1_train = index[0][array[:n_per]]
        array2_train = index[1][array[:n_per]]
        array1_test = index[0][array[n_per:]]
        array2_test = index[1][array[n_per:]]
    else:
        array1_train = np.concatenate((array1_train,index[0][array[:n_per]]))
        array2_train = np.concatenate((array2_train,index[1][array[:n_per]]))
        array1_test = np.concatenate((array1_test,index[0][array[n_per:]]))
        array2_test = np.concatenate((array2_test,index[1][array[n_per:]]))
    Y_train[index[0][array[:n_per]],index[1][array[:n_per]]] = i
    n_sample_train += n_per
    n_sample_test += n_sample-n_per
    
array3 = np.zeros(array1_train.shape)

y_train = one_hot3d(Y_train)
y_train = np.reshape(y_train,(1,row,col,num_class))
X_train = np.reshape(X,(1,row,col,n_band))

mask_train = np.zeros(y_train.shape)
for i in range(num_class):
    mask_train[:,:,:,i] = np.sum(y_train,-1)
    
time1 = time.time()
num_epoch = 4000

histloss = np.zeros((num_epoch,2))

for i in range(num_epoch):

    train_accuracy = 0
    train_loss = 0
    
    sess.run(train_step,feed_dict={x:X_train,y_:y_train,idx1:array1_train,idx2:array2_train,idx3:array3})
    
    if (i+1)%100==0:
        train_accuracy,train_loss = sess.run([accuracy,cross_entropy],feed_dict={x:X_train,y_:y_train,idx1:array1_train,idx2:array2_train,idx3:array3})
       
        histloss[i,:] = train_accuracy,train_loss
    
        print("epoch %d, train_accuracy %g, train_loss %g"%(i+1, histloss[i,0],histloss[i,1]))

            
label = np.squeeze(sess.run(y_label,feed_dict={x:X_train,y_: y_train,idx1:array1_train,idx2:array2_train,idx3:array3}))
prob = np.squeeze(sess.run(y_prob,feed_dict={x:X_train,y_: y_train,idx1:array1_train,idx2:array2_train,idx3:array3}))    

y_test = Y[array1_test,array2_test]-1
y_pred = label[array1_test,array2_test]

OA,kappa,ProducerA = CalAccuracy(y_pred,y_test)

FCN_joint[:num_class] = ProducerA
FCN_joint[-3] = OA
FCN_joint[-2] = kappa

time2 = time.time()    
FCN_joint[-1] = time2 - time1

print("Running time: %g"%(FCN_joint[-1]))

img = DrawResult(np.reshape(label+1,-1),2)
plt.imsave(resultpath+'FCNJoint'+'_'+repr(int(OA*10000))+'.png',img)

time1 = time.time()
CNNMap = np.squeeze(sess.run(h_SS,feed_dict={x:X_train,y_: y_train,idx1:array1_train,idx2:array2_train,idx3:array3}))
#np.squeeze：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
n_feature = CNNMap.shape[-1]
PC_num = 5
image = featureNormalize(PCANorm(CNNMap.reshape(row*col, n_feature),PC_num),1).reshape(row,col,PC_num)     
softmax = prob.transpose((2, 0, 1))

unary = softmax_to_unary(softmax)
#之前先对图片使用训练好的网络预测得到最终经过softmax函数得到的分类结果，这里需要将这个概率结果转成一元
unary = np.ascontiguousarray(unary)
#ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

#下面就是CRF的部分
d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 17)
#第一个参数是待分割图像的尺寸，第二个参数是类别
d.setUnaryEnergy(unary)
#将之前softmax转换的一元势添加到CRF中
feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])
#创建二元高斯势的Util函数（即创建与颜色无关的功能将他们添加到CRF中去），这适用于所有的图像尺寸；第一个参数是每个维度的比例因子，第二个参数是CRF的形状

d.addPairwiseEnergy(feats, compat=1,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

feats = create_pairwise_bilateral(sdims=(30, 30), schan=[5],
                                    img=image, chdim=2)
#创造二元双边势的Util函数（即创建与颜色无关的功能将他们添加到CRF中去），这适用于所有的图像尺寸；第一个参数是每个维度的比例因子，第二个参数是图像中每个通道的比例因子，第三个参数是输入图像，指定了通道维度在图像中的位置。例如，' chdim=2 '用于大小为(240,300,3)的RGB图像,指定维度值3放在索引2的位置处。如果图像没有通道尺寸(例如只有一个通道)，则使用' chdim=-1
d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
#将一元势和二元势结合起来就能够比较全面地去考量像素之间的关系，并得出优化后的结果。
#用5次迭代进行推理最简单的方法是:
Q = d.inference(5)

res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

#以下部分就是得到结果验证精度
y_test = Y[array1_test,array2_test]-1
y_pred = res[array1_test,array2_test]

OA,kappa,ProducerA = CalAccuracy(y_pred,y_test)

FCN_crf[:num_class] = ProducerA
FCN_crf[-3] = OA
FCN_crf[-2] = kappa

time2 = time.time()    
FCN_crf[-1] = FCN_joint[-1] + time2 - time1

crf_img = DrawResult(np.reshape(res+1,-1),2)
plt.imsave(resultpath+'FCNcrf'+'_'+repr(int(OA*10000))+'.png',crf_img)
    
FCN_joint[:-1] = FCN_joint[:-1]*100
FCN_crf[:-1] = FCN_crf[:-1]*100

sio.savemat(resultpath+'FCN_jointSCUT.mat', {'FCN_joint': FCN_joint})
sio.savemat(resultpath+'FCN_crfSCUT.mat', {'FCN_crf': FCN_crf})