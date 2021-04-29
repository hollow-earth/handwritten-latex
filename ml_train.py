import tensorflow as tf
from numpy import load, argmax, reshape, shape
from cv2 import cv2

processedPath = "./Training Set/processed/"

trainingSet = load(processedPath + "trainingSet.npy")
trainingLabels = load(processedPath + "trainingLabels.npy")
testingSet = load(processedPath + "testingSet.npy")
testingLabels = load(processedPath + "testingLabels.npy")

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(100, 100)), 
	tf.keras.layers.Dense(128, activation='relu'), 
	tf.keras.layers.Dropout(0.2), 
	tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(trainingSet, trainingLabels, epochs=100)
model.evaluate(testingSet, testingLabels)
model.save("./Model/")

image = cv2.imread("Test images/0/0_1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (100,100), interpolation= cv2.INTER_NEAREST)
image = 255-image
image = reshape(image, (1,100,100))

pred = model.predict(image)
pred = argmax(pred)
print(pred)
#label = np.argmax(labelsTesting)


"""trainingSet = reshape(trainingSet, (trainingSet.shape[0], 200, 200, 1))
testingSet = reshape(testingSet, (testingSet.shape[0], 200, 200, 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(2, data_format="channels_last", kernel_size = 12 , padding="same", activation=tf.nn.relu, input_shape=(200,200,1)),
	tf.keras.layers.Flatten(input_shape=(200, 20)), 
	#tf.keras.layers.Dense(128, activation='relu'), 
	tf.keras.layers.Dropout(0.3), 
	tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(trainingSet, trainingLabels, epochs=50)
model.evaluate(testingSet, testingLabels)
model.save("./Model/")

image = cv2.imread("Test images/0/0_1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = 255-image
image = reshape(image, (1,200,200,1))"""