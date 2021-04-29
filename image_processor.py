import numpy as np
from cv2 import cv2
from os import path, mkdir, listdir, rename, walk
from random import randint

originalsPath = "./Training Set/originals/"
processedPath = "./Training Set/processed/"
trainingPath = "./Training Set/training/"
testingPath = "./Training Set/testing/"

digitsDict = dict((str(i),i) for i in range(0,10))
alphabetDict = dict([([letter for letter in "abcdefghijklmnopqrstuvwxxyz"][i-10], i) for i in range(10,37)])
dataDictExt = {"!":37, "(":38, ")":39, ",":40, "[":41, "]":42, "{":43, "}":44, "add":45, "alpha":46, "beta":47, "cos":48, "Delta":49, "div":50, "eq":51,
                  "exists":52, "forall":53, "forward_slash":54, "gamma":55, "geq":56, "gt":57, "in":58, "infty":59, "int":60, "lambda":61, "ldots":62, "leq":63,
                  "lim":64, "log":65, "lt":66, "mu":67, "neq":68, "phi":69, "pi":70, "pm":71, "rightarrow":72, "sigma":73, "sin":74, "sqrt":75, "sub":76, 
                  "sum":77, "tan":78, "theta":79, "times":80}
tensorDict = dict(digitsDict, **alphabetDict, **dataDictExt)

def Splitter():
    for folder in listdir(processedPath):
        if path.isdir(trainingPath + folder) == False:
            mkdir(trainingPath + folder)
        if path.isdir(testingPath + folder) == False:
            mkdir(testingPath + folder)

            files = listdir(processedPath + folder)
            i = round(0.25 * len(files))
            x = 0
            while i > 0:        # This is where we split for the testing set
                randomNumber = randint(0, len(files)-1-x)
                rename(processedPath + folder + "/" + files[randomNumber], testingPath + folder + "/" + files[randomNumber])
                files.pop(randomNumber)
                i -= 1
                x += 1
            
            for leftoverFiles in files:
                rename(processedPath + folder + "/" + leftoverFiles, trainingPath + folder + "/" + leftoverFiles)

def GridProcessor(): # This function takes my originals as inputs, then outputs images ready for the ML to use
    for files in listdir(originalsPath):
        folderName = files[0]
        image = cv2.imread(originalsPath + files)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = 255-image
        cv2.imshow("img", image)
        cv2.waitKey(0)
        counter = 0
        if path.isdir(processedPath + folderName) == False:
            mkdir(processedPath + folderName)
        for xCoord in range(4, 2022, 202):
            for yCoord in range(4, 2022, 202):
                splicedImage = image[xCoord:xCoord+200, yCoord:yCoord+200]
                cv2.imwrite(processedPath + folderName + "/" + folderName + "_" + str(counter) + ".png", splicedImage)
                counter += 1

def NumpyProcessor():
    numberOriginals = len(listdir(originalsPath))

    lengthTrainingSet = 75*numberOriginals
    lengthTestingSet = 25*numberOriginals

    trainingSet = np.zeros((lengthTrainingSet, 100, 100), dtype=np.uint8)
    trainingLabels = np.zeros((lengthTrainingSet), dtype=np.uint8)
    testingSet =  np.zeros((lengthTestingSet, 100, 100), dtype=np.uint8)
    testingLabels = np.zeros((lengthTestingSet), dtype=np.uint8)

    globalCounter = 0

    for files in listdir(originalsPath):
        answerKey = tensorDict[files[0]]
        image = cv2.imread(originalsPath + files)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = 255-image
        
        globalCounter += 1
        localCounter = 0

        for xCoord in range(4, 2022, 202):
            for yCoord in range(4, 2022, 202):
                splicedImage = image[xCoord:xCoord+200, yCoord:yCoord+200]
                splicedImage = cv2.resize(splicedImage, (100,100), interpolation= cv2.INTER_NEAREST)
                if localCounter <= 74:
                    i = localCounter + 75 * (globalCounter-1)
                    trainingSet[i] = splicedImage
                    trainingLabels[i] = answerKey
                else:
                    i = localCounter - 75 + 25 * (globalCounter-1)
                    testingSet[i] = splicedImage
                    testingLabels[i] = answerKey
                localCounter += 1

    np.save(processedPath + "trainingSet", trainingSet)
    np.save(processedPath + "trainingLabels", trainingLabels)
    np.save(processedPath + "testingSet", testingSet)
    np.save(processedPath + "testingLabels", testingLabels)


#GridProcessor() # This is for graphical output. Otherwise, the images are saved as numpy arrays
#Splitter() # This is for splitting images 75%-25% in folders.
NumpyProcessor()