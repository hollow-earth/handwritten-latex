import tensorflow as tf
from numpy import load, argmax, reshape, shape
from cv2 import cv2
from os import listdir

processedPath = "./Training Set/processed/"
testImagesPath = "./Test Images/"

digitsDict = dict((str(i),i) for i in range(0,10))			# Might wanna clean up a bit here
alphabetDict = dict([([letter for letter in "abcdefghijklmnopqrstuvwxxyz"][i-10], i) for i in range(10,37)])
dataDictExt = {"!":37, "(":38, ")":39, ",":40, "[":41, "]":42, "{":43, "}":44, "add":45, "alpha":46, "beta":47, "cos":48, "Delta":49, "div":50, "eq":51,
                  "exists":52, "forall":53, "forward_slash":54, "gamma":55, "geq":56, "gt":57, "in":58, "infty":59, "int":60, "lambda":61, "ldots":62, "leq":63,
                  "lim":64, "log":65, "lt":66, "mu":67, "neq":68, "phi":69, "pi":70, "pm":71, "rightarrow":72, "sigma":73, "sin":74, "sqrt":75, "sub":76, 
                  "sum":77, "tan":78, "theta":79, "times":80}
capitalDict = dict([([letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"][i-81], i) for i in range(81,107)])
tensorDict = dict(digitsDict, **alphabetDict, **dataDictExt, **capitalDict)

def reverseDict(inputValue):			# This has to be changed! Make a reverse dictionary
	for i in tensorDict:
		if tensorDict[i] == inputValue:
			return i

trainingSet = load(processedPath + "trainingSet.npy")
trainingLabels = load(processedPath + "trainingLabels.npy")
testingSet = load(processedPath + "testingSet.npy")
testingLabels = load(processedPath + "testingLabels.npy")

trainingSet = reshape(trainingSet, (trainingSet.shape[0], 100, 100, 1)) #channels_last
testingSet = reshape(testingSet, (testingSet.shape[0], 100, 100, 1))
#trainingSet = trainingSet.astype("float32")
#testingSet = trainingSet.astype("float32")    

model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, input_shape = (100,100,1), padding="same", activation=tf.nn.relu),
	tf.keras.layers.MaxPooling2D(pool_size=2),
	tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding="same", activation=tf.nn.relu),
	tf.keras.layers.MaxPooling2D(pool_size=2),
	tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding="same", activation=tf.nn.relu),
	tf.keras.layers.MaxPooling2D(pool_size=2),
	tf.keras.layers.Dense(units = 1024, activation=tf.nn.relu),
	tf.keras.layers.Flatten()
	#tf.keras.layers.Dropout(0.4),
	#tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(trainingSet, trainingLabels, epochs=50)
model.evaluate(testingSet, testingLabels)
model.save("./Model/")

#model = tf.keras.models.Sequential([
#	tf.keras.layers.Flatten(input_shape=(100, 100)), 
#	tf.keras.layers.Dense(128, activation='relu'), 
#	tf.keras.layers.Dropout(0.2), 
#	tf.keras.layers.Dense(10, activation='softmax')])
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.fit(trainingSet, trainingLabels, epochs=50)
#model.evaluate(testingSet, testingLabels)
#model.save("./Model/")


#model = tf.keras.models.Sequential([
#	tf.keras.layers.Conv2D(filters = 32, kernel_size = [7,7], input_shape = (100,100,1), padding="same", activation=tf.nn.relu),
#	tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
#	tf.keras.layers.Conv2D(filters = 64, kernel_size = [7,7], input_shape = (100,100,1), padding="same", activation=tf.nn.relu),
#	tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
#	tf.keras.layers.Conv2D(filters = 128, kernel_size = [7,7], input_shape = (100,100,1), padding="same", activation=tf.nn.relu),
#	tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
#	tf.keras.layers.Dense(units = 2048, activation=tf.nn.relu),
#	tf.keras.layers.Dropout(0.4),
#	tf.keras.layers.Dense(10, activation='softmax')
#])
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.fit(trainingSet, trainingLabels, epochs=50)
#model.evaluate(testingSet, testingLabels)
#model.save("./Model/")


for folder in listdir(testImagesPath):
	for img in listdir(testImagesPath + folder):
		image = cv2.imread(testImagesPath + folder + "/" + img)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (100,100), interpolation= cv2.INTER_NEAREST)
		image = 255-image
		image = reshape(image, (1,100,100,1))

		pred = model.predict(image)
		pred = argmax(pred)
		pred = reverseDict(pred)
		print("The prediction is: {prediction}. The answer is: {answer}.".format(prediction=pred, answer=img[0]))
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

image = cv2.imread("Test images/0/0_2.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = 255-image
image = reshape(image, (1,200,200,1))"""