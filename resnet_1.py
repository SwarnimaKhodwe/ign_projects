import cv2
import numpy as np
from numpy import zeros
import math
import tensorflow as tf

class resnet_2:

    def read(self,name):
        self.name = name
        self.img = cv2.imread(self.name,-1)  #read the image in original format

    def preprocess(self):
    
        if len(self.img.shape) == 2:   #check weather the image is grayscale or not
            print("Invalid image.")  
            return 0

        h = self.img.shape[0]  #calculate the height of original image
        w = self.img.shape[1]  #calculate the width of original image

        if h != 256 or w != 256:  #check if any of the dimenssion is not equal to 256
            self.img = cv2.resize(self.img,(256,256),interpolation = cv2.INTER_AREA)  #resize the image such that height=256, width=256

        self.img1 = self.img[16:240,16:240]  #take centre crop the image 

        h1 = self.img1.shape[0]  #calculate the height of cropped image 
        w1 = self.img1.shape[1]  #calculate the width of cropped image

        R = self.img1[:,:,2]  #calculate the Red pixel value
        G = self.img1[:,:,1]  #calculate the Green pixel value
        B = self.img1[:,:,0]  #calculate the Blue pixel value

        # declare three matrices of size equal to the cropped image
        r = zeros([h1,w1])  
        g = zeros([h1,w1])
        b = zeros([h1,w1])

        #calculate the MEAN value of Red, Green, Blue pixels
        for i in range (0,h1):
            for j in range (0,w1):
                r[i][j] = R[i][j]-123.68
                g[i][j] = G[i][j]-116.78
                b[i][j] = B[i][j]-103.94

        #merge the matrices r,g,b to get the new image 
        self.img2 = cv2.merge((r,g,b))

        self.img2 = np.expand_dims(self.img2,1)  #add new dimenssion
        self.img2 = np.rollaxis(self.img2, 1, 0)  #swaap the position 0 with 1 to get the required format 
        

        return self.img2
