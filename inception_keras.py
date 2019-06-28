
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import decode_predictions
import numpy as np
import cv2
from inception_1 import inception_2

model = InceptionV3(weights='imagenet')

rk = inception_2()
rk.read('abc.jpg')
x = rk.preprocess()

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])

