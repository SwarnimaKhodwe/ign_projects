from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.applications.mobilenet import decode_predictions
import numpy as np
import cv2
import matplotlib.pyplot as plt
from resnet_1 import resnet_2

model = ResNet50(weights='imagenet')
#img_path="car.jpg"
rk = resnet_2()
rk.read('s.jpg')

x = rk.preprocess()

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
