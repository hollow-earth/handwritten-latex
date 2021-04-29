import tensorflow as tf
import numpy as np
from cv2 import cv2
from mnist import MNIST
from wget import download
from os import path, mkdir, listdir, remove, walk

trainingPath = "./Training Sets/"
hmsdPath = "./Training Sets/Handwritten math symbols dataset/"
HMSPath = "./Training Sets/Handwritten Math Symbols/"

digitsDict = dict((str(i),i) for i in range(0,10))
alphabetDict = dict([([letter for letter in "abcdefghijklmnopqrstuvwxxyz"][i-10], i) for i in range(10,37)])
dataDictExt = {"!":37, "(":38, ")":39, ",":40, "[":41, "]":42, "{":43, "}":44, "add":45, "alpha":46, "beta":47, "cos":48, "Delta":49, "div":50, "eq":51,
                  "exists":52, "forall":53, "forward_slash":54, "gamma":55, "geq":56, "gt":57, "in":58, "infty":59, "int":60, "lambda":61, "ldots":62, "leq":63,
                  "lim":64, "log":65, "lt":66, "mu":67, "neq":68, "phi":69, "pi":70, "pm":71, "rightarrow":72, "sigma":73, "sin":74, "sqrt":75, "sub":76, 
                  "sum":77, "tan":78, "theta":79, "times":80}
tensorDict = dict(digitsDict, **alphabetDict, **dataDictExt)

def trainML(trainingSet, trainingLabels, testingSet, testingLabel):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(2, data_format="channels_last", kernel_size = 12 , padding="same", activation=tf.nn.relu, input_shape=(48,48,1)),
	tf.keras.layers.Flatten(input_shape=(48, 48)), 
	#tf.keras.layers.Dense(128, activation='relu'), 
	tf.keras.layers.Dropout(0.3), 
	tf.keras.layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainingSet, trainingLabels, epochs=50)
    model.evaluate(testingSet, testingLabel)
    model.save("./Model/")

def progressBar(currentNumber, totalNumber, currentTask):
    if currentNumber % 50 == 0 or currentNumber == totalNumber:                                             # The value should be changed to be approximately 1%
        progress = round(currentNumber/totalNumber * 100)
        bar = "[" + "="*progress + "."*(100-progress) + "]" + "   " + str(progress) + "%"
        print(bar, "   Current task: {task}".format(task=currentTask), end="\r")


def HMSProcessor():
    numberFilesTraining = 0
    numberFilesTesting = 0
    for _, _, files in walk(HMSPath + "training/"):
        numberFilesTraining += len(files)
    for _, _, files in walk(HMSPath + "testing/"):
        numberFilesTesting += len(files)   
    trainingSetHMS = np.empty((numberFilesTraining, 48, 48, 1), dtype=np.uint8)
    trainingLabelsHMS = np.empty((numberFilesTraining), dtype=np.uint8)
    testingSetHMS = np.empty((numberFilesTesting, 48, 48, 1), dtype=np.uint8)
    testingLabelsHMS = np.empty((numberFilesTesting), dtype=np.uint8)

    for sets in ["training/", "testing/"]:
        counter = 0
        for folder in listdir(HMSPath + sets):
            for files in listdir(HMSPath + sets + folder):
                img = cv2.imread(HMSPath + sets + folder + "/" + files)
                img = cv2.resize(img, (46,46), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.threshold(img,175,255,cv2.THRESH_BINARY_INV)[1]
                img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
                for i in [0,1,2,3,42,43,44,45]:
                    if img[i][0] != 0 and img[i][45] != 0:
                        img[i] *= 0
                cv2.imshow("img", img)
                cv2.waitKey(0)
                if sets == "training/":
                    trainingSetHMS[counter,:,:,0] = img
                    trainingLabelsHMS[counter] = tensorDict[folder]
                    progressBar(counter, numberFilesTraining, "Processing the training set for HMS    ")
                if sets == "testing/":
                    testingSetHMS[counter,:,:,0] = img
                    testingLabelsHMS[counter] = tensorDict[folder]
                    progressBar(counter, numberFilesTesting, "Processing the testing set for HMS   ")
                counter += 1

    return trainingSetHMS, trainingLabelsHMS, testingSetHMS, testingLabelsHMS

dataTrainingHMS, labelsTrainingHMS, dataTestingHMS, labelsTestingHMS = HMSProcessor()                 # Untested but so far looks fine
#np.save("./Processed/dataTrainingHmsd.npy", dataTrainingHmsd)
#np.save("./Processed/labelsTrainingHmsd.npy", labelsTrainingHmsd)
#np.save("./Processed/dataTestingHmsd.npy", dataTestingHmsd)
#np.save("./Processed/labelsTestingHmsd.npy", labelsTestingHmsd)

trainML(dataTrainingHMS, labelsTrainingHMS, dataTestingHMS, labelsTestingHMS)