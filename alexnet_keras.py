import urllib.request
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes
import glob
from alexnet_1 import alexnet_2

rk = alexnet_2()

dropoutPro = 1
classNum = 1000
skip = []

x = tf.placeholder("float", [1, 227, 227, 3])
 
 
model = alexnet.alexNet(x, dropoutPro, classNum, skip)
score = model.fc3
softmax = tf.nn.softmax(score)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.loadModel(sess) #Load the model

       
    rk.read('abc.jpg')
    test = rk.preprocess()

    maxx = np.argmax(sess.run(softmax, feed_dict = {x: test}))#here goes the preprocessed image
    res = caffe_classes.class_names[maxx] #find the max probility
    print(res)
