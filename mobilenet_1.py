import cv2
import numpy as np
from numpy import zeros

class mobilenet_2:
    
    def read(self,name):
        self.name = name
        self.img = cv2.imread(self.name,-1)  #read the image in original format

    def preprocess(self):
        if len(self.img.shape) == 2:  ##check weather the image is grayscale or not
            print("Invalid image.")
            return 0
          
        self.img1 = cv2.resize(self.img,(256,256), interpolation=cv2.INTER_LINEAR)  ##resize the image such that height=256, width=256
           
        self.img2 = self.img1[0:224,0:224]  #take centre crop the image 
              
        h = self.img2.shape[0]  #calculate the height of img2
        w = self.img2.shape[1]  #calculate the width of img2
          
        R = self.img2[:,:,2]  #calculate the Red pixel value
        G = self.img2[:,:,1]  #calculate the Green pixel value
        B = self.img2[:,:,0]  #calculate the Blue pixel value
              
        # declare three matrices of size equal to the cropped image
        r = zeros([h,w])  
        g = zeros([h,w])
        b = zeros([h,w])

        #calculate the MEAN value of Red, Green, Blue pixels
        for i in range (0,h):
            for j in range (0,w):
                r[i][j] = R[i][j]-123.68
                g[i][j] = G[i][j]-116.78
                b[i][j] = B[i][j]-103.94

        #merge the matrices r,g,b to get the new image 
        self.img3 = cv2.merge((r,g,b))

        h1 = self.img3.shape[0]  #calculate the height of img3
        w1 = self.img3.shape[1]  #calculate the width of img3

        #Normalize the pixels to range of +/-1.0
        for i in range (0,h1):
            for j in range (0,w1):
                self.img3[i][j]= self.img3[i][j]/255

        self.img3 = np.expand_dims(self.img3,1)  #add new dimenssion
        self.img3 = np.rollaxis(self.img3, 1, 0)  #swaap the position 0 with 1 to get the required format 

        return self.img3

    
