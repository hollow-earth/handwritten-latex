#import tensorflow as tf
from cv2 import cv2
from numpy import argmax
from numpy import shape
#model = tf.keras.models.load_model("C:/Coding/Github Projects/handwritten-latex/Model/")


def processImage(image, thresholdValue):
  img = cv2.imread(image)
  grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  denoised = cv2.fastNlMeansDenoising(grayScale,h=5)
  result = 255-denoised
  processedImage = cv2.threshold(result, thresholdValue, 255, cv2.THRESH_BINARY)[1]
  processedImageClone = cv2.cvtColor(processedImage, cv2.COLOR_GRAY2BGR)

  thresh = cv2.threshold(processedImage,thresholdValue,255,0)[1]
  contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

  for i in range(0,shape(contours)[0]):
    x,y,w,h = cv2.boundingRect(contours[i])
    img = cv2.rectangle(processedImageClone,(x,y),(x+w,y+h),(0,255,0),2)

  return processedImageClone
  #cv2.imshow("image", processedImageClone)
  #cv2.waitKey(0)


"""for i in range(0,8):
  image = cv2.imread("C:/Coding/Github Projects/handwritten-latex/Test images/"+str(i)+".jpg", 0)/255
  image = image.reshape(1,28,28)
  pred = model.predict(image) 
  pred = argmax(pred)
  print(pred) """