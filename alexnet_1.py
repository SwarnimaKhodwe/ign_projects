import cv2
import numpy as np
from numpy import zeros

class alexnet_2:

    def read(self,name):
        self.name = name
        self.img = cv2.imread(self.name,-1)  #read the image in original format

    def preprocess(self):
    
        if len(self.img.shape) == 2:   #check weather the image is grayscale or not
            print("Invalid image.")  
            return 0

        #self.img = np.array([104, 117, 124], np.float)

        self.img1 = cv2.resize(self.img.astype(float), (227, 227))

        h = self.img1.shape[0]  #calculate the height of cropped image 
        w = self.img1.shape[1]  #calculate the width of cropped image

        R = self.img1[:,:,2]  #calculate the Red pixel value
        G = self.img1[:,:,1]  #calculate the Green pixel value
        B = self.img1[:,:,0]  #calculate the Blue pixel value

        # declare three matrices of size equal to the cropped image
        r = zeros([h,w])  
        g = zeros([h,w])
        b = zeros([h,w])

        #calculate the MEAN value of Red, Green, Blue pixels
        for i in range (0,h):
            for j in range (0,w):
                r[i][j] = R[i][j]-104
                g[i][j] = G[i][j]-117
                b[i][j] = B[i][j]-124

        #merge the matrices r,g,b to get the new image 
        self.img2 = cv2.merge((r,g,b))

        self.img3 = self.img2.reshape((1, 227, 227, 3))

        return self.img3


        
