import cv2
from numpy import zeros
import numpy as np
import math

class inception_2:
    def read(self,name):
        self.name = name
        self.img = cv2.imread(self.name,-1)  #read the image in original format

    def preprocess(self):
        if len(self.img.shape) == 2:  #check weather the image is grayscale or not
            print("Invalid image.")
            return 0


        h = self.img.shape[0]  #calculate the height of original image
        w = self.img.shape[1]  #calculate the width of original image

        if h !=299 or w != 299:
            self.img1 = cv2.resize(self.img,(299,299),interpolation = cv2.INTER_LINEAR)


        h1 = self.img1.shape[0]  #calculate the height of original image
        w1 = self.img1.shape[1]  #calculate the width of original image
        
        R = self.img1[:,:,2]  #calculate the Red pixel value
        G = self.img1[:,:,1]  #calculate the Green pixel value
        B = self.img1[:,:,0]  #calculate the Blue pixel value

        # declare three matrices of size equal to the img1
        r = zeros([h1,w1])  
        g = zeros([h1,w1])
        b = zeros([h1,w1])
        
        for i in range (0,h1):
            for j in range (0,w1):
                r[i][j] = (((float(R[i][j])/255)-0.5)*2)
                g[i][j] = (((float(G[i][j])/255)-0.5)*2)
                b[i][j] = (((float(B[i][j])/255)-0.5)*2)
                
        #merge the matrices r,g,b to get the new image 
        self.img2 = cv2.merge((r,g,b))


        self.img2 = np.expand_dims(self.img2,1)  #add new dimenssion
        self.img2 = np.rollaxis(self.img2, 1, 0)  #swap the position 0 with 1 to get the required format
       
        return self.img2


       
