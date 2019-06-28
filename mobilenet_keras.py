from keras.applications.mobilenet import MobileNet
#from keras.preprocessing import image
#from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from mobilenet_1 import mobilenet_2
from keras.applications.mobilenet import decode_predictions

model = MobileNet(weights='imagenet')

rk = mobilenet_2()
rk.read('car.jpg')
x = rk.preprocess()

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])

