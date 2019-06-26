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
'''img = image.load_img(img_path, target_size=(224, 224))  # read image in PIL format.
x_original = image.img_to_array(img)                             # simply converting into numpy array. no processing.
x_original = np.expand_dims(x_original, axis=0)          # just converting into 1,224,224,3.

x_original = preprocess_input(x_original)
print("SHAPE original:  ",x_original.shape)
print(type(x_original))
print(x_original)
'''
x = rk.preprocess()
#print("SHAPE rk:  ",x.shape)
#print(type(x))
#print(x)
preds = model.predict(x)
#print("raw predicts:",preds)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print("type:",type(preds))
#print("raw predicts:",np.argmax(preds))
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
