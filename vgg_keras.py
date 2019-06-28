from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import decode_predictions
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vggnet_1 import vggnet_2

model = VGG16(weights='imagenet')

rk = vggnet_2()
rk.read('car.jpg')

x = rk.preprocess()

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

