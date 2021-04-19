import tensorflow as tf
import numpy as np
from cv2 import cv2
from mnist import MNIST

mndata = MNIST("C:/Users/Sam/Desktop/Training Sets/MNIST/")

imagesTraining, labelsTraining = mndata.load_training()
imagesTesting, labelsTesting = mndata.load_testing()

imagesTraining, labelsTraining = np.array(imagesTraining)/255, np.array(labelsTraining)
imagesTesting, labelsTesting = np.array(imagesTesting)/255, np.array(labelsTesting)
imagesTraining = np.reshape(imagesTraining, (60000,28,28))
imagesTesting = np.reshape(imagesTesting, (10000,28,28))

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(imagesTraining, labelsTraining, epochs=150)
model.evaluate(imagesTesting, labelsTesting)

model.save("C:/Coding/Github Projects/handwritten-latex/Model/")

#pred = model.predict(image) 
#pred = np.argmax(pred)
#label = np.argmax(labelsTesting)

#print(pred) 
#print(label)
