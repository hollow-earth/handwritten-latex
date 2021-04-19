import tensorflow as tf
from cv2 import cv2
from numpy import argmax

model = tf.keras.models.load_model("C:/Coding/Github Projects/handwritten-latex/Model/")

for i in range(0,8):
  image = cv2.imread("C:/Coding/Github Projects/handwritten-latex/Test images/"+str(i)+".jpg", 0)/255
  image = image.reshape(1,28,28)
  pred = model.predict(image) 
  pred = argmax(pred)
  print(pred) 