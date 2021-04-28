import tensorflow as tf
import numpy as np
from cv2 import cv2
from mnist import MNIST
from wget import download
from os import path, mkdir, listdir, remove, walk
import gzip

mnistPath = "./Training Sets/MNIST/"
kagglePath = "./Training Sets/Kaggle/"

dataDictionary = {"add":"+", "sub":"-", "times":"\\times", "div":"\\div", "dec":".", "forward_slash":"/", "pm":"\\pm", "sqrt":"\\sqrt", "sum":"\\sum",
                  "{":"\\lbrace", "}":"\\rbrace", "ldots":"\\ldots",
                  "eq":"=", "geq":"\\geq", "gt":">", "leq":"\\leq", "lt":"<", "neq":"\\neq", 
                  "alpha":"\\alpha", "beta":"\\beta", "Delta":"\\Delta", "gamma":"\\gamma", "lambda":"\\lambda",  "mu":"\\mu", "pi":"\\pi", "phi":"\\phi", "sigma":"\\sigma",
                        "theta":"\\theta",
                  "cos":"\\cos", "sin":"\\sin", "tan":"\\tan", "log":"\\log",
                  "exists":"\\exists", "forall":"\\forall", "in":"\\in",
                  "infty":"\\infty",
                  "int":"\\int", "lim":"\\lim",
                  "rightarrow":"\\rightarrow"}

digitsDictionary = dict((str(i),i) for i in range(0,10))
alphabetDictionary = dict([([letter for letter in "abcdefghijklmnopqrstuvwxxyz"][i-10], i) for i in range(10,37)])
dataRefDicitionary = dict([(list(dataDictionary.values())[i-37], i) for i in range(37, 37+len(dataDictionary))])
additionalDictionary = {"!":76, "(":77, ")":78, "[":79, "]":80, ",":81}
tensorDictionary = dict(digitsDictionary, **alphabetDictionary, **dataRefDicitionary, **additionalDictionary)


def trainML(trainingSet, trainingLabels, testingSet, testingLabel):
    model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(128, 128)), 
	tf.keras.layers.Dense(128, activation='relu'), 
	tf.keras.layers.Dropout(0.2), 
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
        if currentNumber == totalNumber:
            print("\n")

def MNISTDownload():
    mnistFiles = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    mnistURL = "http://yann.lecun.com/exdb/mnist/"

    if path.isdir(mnistPath) == False:
        mkdir(mnistPath)
    
    for files in mnistFiles:    
        if path.isfile(mnistPath + files) == False and path.isfile(mnistPath + files[:-3]) == False:        # If file and uncompressed file don't exist, download it
            download(mnistURL + files, mnistPath + files)
        
        if path.isfile(mnistPath + files[:-3]) == False and path.isfile(mnistPath + files) == True:         # If uncompressed file doesn't exist but file does, 
            with gzip.open(mnistPath + files, 'rb') as s_file, open(mnistPath + files[:-3], 'wb') as d_file:# uncompress then delete
                d_file.write(s_file.read())
        
        if path.isfile(mnistPath + files[:-3]) == True and path.isfile(mnistPath + files) == True:          # If for some reason file was uncompressed but not deleted,
            remove(mnistPath + files)                                                                       # then delete the uncompressed file

def HMSProcessor():                                                                                         # Images aren't processed whatsoever, lines must be removed,
    HMSPath = "./Training Sets/Handwritten Math Symbols/"                                                   # images must be resized (possibly centered too) and 
    numberFilesTraining = 0                                                                                 # colors have to be inverted
    numberFilesTesting = 0
    for _, _, files in walk(HMSPath + "training/"):                                                         # Couldn't find a better way to count the number of files
        numberFilesTraining += len(files)                                                                   # In the directory + all subdirectories
    for _, _, files in walk(HMSPath + "testing/"):
        numberFilesTesting += len(files)   
    trainingSetHMS = np.empty((numberFilesTraining, 128, 128), dtype=np.uint8)
    trainingLabelsHMS = np.empty((numberFilesTraining), dtype="<U6")
    testingSetHMS = np.empty((numberFilesTesting, 128, 128), dtype=np.uint8)
    testingLabelsHMS = np.empty((numberFilesTesting), dtype="<U6")

    for sets in ["training/", "testing/"]:
        counter = 0
        for folder in listdir(HMSPath + sets):
            for files in listdir(HMSPath + sets + folder):
                img = cv2.imread(HMSPath + sets + folder + "/" + files)
                img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
                grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(grayScale,h=5)
                result = 255-denoised
                for i in [0,1,2,3,122,123,124,125,126,127]:                                                 # This particular set has sometimes horizontal lines that 
                    if result[i][0] != 0 and result[i][127] != 0:                                           # span an entire row. If it finds such a row (both endpoint)
                        result[i] *= 0                                                                      # then we set that particular row's value to 0
                    if i < 4 and np.sum(result[i]) != 0:                                                    # This is set-specific
                        result[i] *= 0
                if sets == "training/":
                    trainingSetHMS[counter] = result
                    if folder not in dataDictionary:
                        trainingLabelsHMS[counter] = str(folder)
                    else:
                        trainingLabelsHMS[counter] = dataDictionary[folder]
                    progressBar(counter, numberFilesTraining, "Processing the training set for HMS    ")
                if sets == "testing/":
                    testingSetHMS[counter] = result
                    if folder not in dataDictionary:
                        testingLabelsHMS[counter] = str(folder)
                    else:
                        testingLabelsHMS[counter] = dataDictionary[folder]
                    progressBar(counter, numberFilesTesting, "Processing the testing set for HMS   ")
                counter += 1

    return trainingSetHMS, trainingLabelsHMS, testingSetHMS, testingLabelsHMS

def HmsaddProcessor():
    HmsadddPath = "./Training Sets/Handwritten math symbol and digit dataset/"                                          # images must be resized (possibly centered too) and 
    numberFilesTraining = 0                                                                                 # colors have to be inverted
    numberFilesTesting = 0
    for _, _, files in walk(HmsadddPath + "training/"):                                                        # Couldn't find a better way to count the number of files
        numberFilesTraining += len(files)                                                                   # In the directory + all subdirectories
    for _, _, files in walk(HmsadddPath + "testing/"):
        numberFilesTesting += len(files)   
    trainingSetHmsadd = np.empty((numberFilesTraining, 128, 128), dtype=np.uint8)
    trainingLabelsHmsadd = np.empty((numberFilesTraining), dtype="<U6")
    testingSetHmsadd = np.empty((numberFilesTesting, 128, 128), dtype=np.uint8)
    testingLabelsHmsadd = np.empty((numberFilesTesting), dtype="<U6")
    
    for sets in ["training/", "testing/"]:
        counter = 0
        for folder in listdir(HmsadddPath + sets):
            for files in listdir(HmsadddPath + sets + folder):
                img = cv2.imread(HmsadddPath + sets + folder + "/" + files)
                img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR)
                grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(grayScale,h=5)
                result = 255-denoised
                for i in [0,1,2,3,122,123,124,125,126,127]:                                                 # This particular set has sometimes horizontal lines that 
                    if result[i][0] != 0 and result[i][127] != 0:                                           # span an entire row. If it finds such a row (both endpoint)
                        result[i] *= 0                                                                      # then we set that particular row's value to 0
                    if i < 4 and np.sum(result[i]) != 0:                                                    # This is set-specific
                        result[i] *= 0
                if sets == "training/":
                    trainingSetHmsadd[counter] = result
                    if folder not in dataDictionary:
                        trainingLabelsHmsadd[counter] = str(folder)
                    else:
                        trainingLabelsHmsadd[counter] = dataDictionary[folder]
                    progressBar(counter, numberFilesTraining, "Processing the training set for Hmsadd")
                if sets == "testing/":
                    testingSetHmsadd[counter] = result
                    if folder not in dataDictionary:
                        testingLabelsHmsadd[counter] = str(folder)
                    else:
                        testingLabelsHmsadd[counter] = dataDictionary[folder]
                    progressBar(counter, numberFilesTesting, "Processing the testing set for Hmsadd")
                counter += 1
    return trainingSetHmsadd, trainingLabelsHmsadd, testingSetHmsadd, testingLabelsHmsadd

def HmsdProcessor():
    HmsdPath = "./Training Sets/Handwritten math symbols dataset/"                                          # images must be resized (possibly centered too) and 
    numberFilesTraining = 0                                                                                 # colors have to be inverted
    numberFilesTesting = 0
    for _, _, files in walk(HmsdPath + "training/"):                                                        # Couldn't find a better way to count the number of files
        numberFilesTraining += len(files)                                                                   # In the directory + all subdirectories
    for _, _, files in walk(HmsdPath + "testing/"):
        numberFilesTesting += len(files)   
    trainingSetHmsd = np.empty((numberFilesTraining, 128, 128), dtype=np.uint8)
    trainingLabelsHmsd = np.empty((numberFilesTraining), dtype="<U11")
    testingSetHmsd = np.empty((numberFilesTesting, 128, 128), dtype=np.uint8)
    testingLabelsHmsd = np.empty((numberFilesTesting), dtype="<U11")
    
    for sets in ["training/", "testing/"]:
        counter = 0
        for folder in listdir(HmsdPath + sets):
            for files in listdir(HmsdPath + sets + folder):
                img = cv2.imread(HmsdPath + sets + folder + "/" + files)
                img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
                grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #denoised = cv2.fastNlMeansDenoising(grayScale,h=5)
                result = 255-grayScale
                if sets == "training/":
                    trainingSetHmsd[counter] = result
                    if folder not in dataDictionary:
                        trainingLabelsHmsd[counter] = str(folder)
                    else:
                        trainingLabelsHmsd[counter] = dataDictionary[folder]
                    progressBar(counter, numberFilesTraining, "Processing the training set for Hmsd  ")
                if sets == "testing/":
                    testingSetHmsd[counter] = result
                    if folder not in dataDictionary:
                        testingLabelsHmsd[counter] = str(folder)
                    else:
                        testingLabelsHmsd[counter] = dataDictionary[folder]
                    progressBar(counter, numberFilesTesting, "Processing the testing set for Hmsd  ")  
                counter += 1
    return trainingSetHmsd, trainingLabelsHmsd, testingSetHmsd, testingLabelsHmsd

def MNISTProcessor():
    MNISTDownload()
    MNISTData = MNIST(mnistPath)

    MNISTImagesTraining, MNISTLabelsTraining = MNISTData.load_training()
    MNISTImagesTesting, MNISTLabelsTesting = MNISTData.load_testing()

    MNISTImagesTraining, MNISTLabelsTraining = np.array(MNISTImagesTraining, dtype=np.uint8), np.array(MNISTLabelsTraining, dtype="<U1")
    MNISTImagesTesting, MNISTLabelsTesting = np.array(MNISTImagesTesting, dtype=np.uint8), np.array(MNISTLabelsTesting, dtype="<U1")
    MNISTImagesTraining = np.reshape(MNISTImagesTraining, (60000, 28, 28))
    MNISTImagesTesting = np.reshape(MNISTImagesTesting, (10000, 28, 28))

    MNISTImagesTrainingProcessed = np.empty((60000,128,128), dtype=np.uint8)
    MNISTImagesTestingProcessed = np.empty((10000,128,128), dtype=np.uint8)

    for image in range(0,np.shape(MNISTImagesTraining)[0]):
        for i in [0,1,26,27]:
            MNISTImagesTraining[image][i] *= 0
            MNISTImagesTraining[image][:,i] *= 0
        MNISTImagesTrainingProcessed[image] = cv2.resize(MNISTImagesTraining[image], (128,128), interpolation=cv2.INTER_AREA)
        progressBar(image, 60000, "Processing the training set for MNIST ")

    for image in range(0, np.shape(MNISTImagesTesting)[0]):
        for i in [0,1,26,27]:
            MNISTImagesTesting[image][i] *= 0
            MNISTImagesTesting[image][:,i] *= 0
        MNISTImagesTestingProcessed[image] = cv2.resize(MNISTImagesTesting[image], (128,128), interpolation=cv2.INTER_AREA)
        progressBar(image, 60000, "Processing the training set for MNIST ")
    
    return MNISTImagesTrainingProcessed, MNISTLabelsTraining, MNISTImagesTestingProcessed, MNISTLabelsTesting

"""def EMNISTProcessor():  # Was too lazy to rewrite the whole thing for EMNIST, so I just gave the EMNIST data files the same name as the MNIST ones
    EMNISTData = MNIST(emnistPath)  # Ended up discarding the idea due to the poor quality of the dataset

    EMNISTImagesTraining, EMNISTLabelsTraining = EMNISTData.load_training()
    EMNISTImagesTesting, EMNISTLabelsTesting = EMNISTData.load_testing()

    trainingLength = np.shape(EMNISTImagesTraining)[0]
    testingLength = np.shape(EMNISTImagesTesting)[0]

    EMNISTImagesTraining, EMNISTLabelsTraining = np.array(EMNISTImagesTraining, dtype=np.uint8), np.array(EMNISTLabelsTraining, dtype=np.str)
    EMNISTImagesTesting, EMNISTLabelsTesting = np.array(EMNISTImagesTesting, dtype=np.uint8), np.array(EMNISTLabelsTesting, dtype=np.str)
    EMNISTImagesTraining = np.reshape(EMNISTImagesTraining, (trainingLength, 28, 28))
    EMNISTImagesTesting = np.reshape(EMNISTImagesTesting, (testingLength, 28, 28))

    EMNISTImagesTrainingProcessed = np.empty((trainingLength,128,128), dtype=np.uint8)
    EMNISTImagesTestingProcessed = np.empty((testingLength,128,128), dtype=np.uint8) 

    for image in range(0, trainingLength):
        #for i in [0,1,26,27]:
        #    MNISTImagesTraining[image][i] *= 0
        #    MNISTImagesTraining[image][:,i] *= 0
        EMNISTImagesTrainingProcessed[image] = cv2.resize(EMNISTImagesTraining[image], (128,128), interpolation=cv2.INTER_AREA)
        progressBar(image, trainingLength, "Processing the training set for EMNIST")
        cv2.imshow("img", EMNISTImagesTrainingProcessed[image])
        print(EMNISTLabelsTraining[image])
        cv2.waitKey(0)

    for image in range(0, testingLength):
        #for i in [0,1,26,27]:
        #    MNISTImagesTesting[image][i] *= 0
        #    MNISTImagesTesting[image][:,i] *= 0
        EMNISTImagesTestingProcessed[image] = cv2.resize(EMNISTImagesTesting[image], (128,128), interpolation=cv2.INTER_AREA)
        progressBar(image, testingLength, "Processing the training set for EMNIST")"""

"""def AZHD(): # Writing a script to extract the data set seems quite complicated, I'll leave it at that for now
    alphabet = []
    alphabetDictionary = {}
    csvFile = np.loadtxt("C:/Coding/Github Projects/handwritten-latex/Training Sets/A-Z Handwritten Alphabet/A_ZHandwrittenData.csv", delimiter=",")
    #print(np.shape(csvFile))
    #print(csvFile[:,0])
    trainingArray = 
    for i in range(0, np.shape(csvFile)[0]):"""

# While this is terrible for storage space (estimated 8-10 GB written and deleted), this kind of checkpoint allows for the program to crash without too many losses.
def finalProcessor():
    if path.isfile("./Processed/finalTrainingSet.npy") == False or path.isfile("./Processed/finalTrainingLabels.npy") == False or\
        path.isfile("./Processed/finalTestingSet.npy") == False or path.isfile("./Processed/finalTestingLabels.npy") == False:
        dataTrainingMNIST, labelsTrainingMNIST, dataTestingMNIST, labelsTestingMNIST = MNISTProcessor()
        np.save("./Processed/dataTrainingMNIST.npy", dataTrainingMNIST)
        np.save("./Processed/labelsTrainingMNIST.npy", labelsTrainingMNIST)
        np.save("./Processed/dataTestingMNIST.npy", dataTestingMNIST)
        np.save("./Processed/labelsTestingMNIST.npy", labelsTestingMNIST)

        dataTrainingHmsd, labelsTrainingHmsd, dataTestingHmsd, labelsTestingHmsd = HmsdProcessor()                 # Untested but so far looks fine
        np.save("./Processed/dataTrainingHmsd.npy", dataTrainingHmsd)
        np.save("./Processed/labelsTrainingHmsd.npy", labelsTrainingHmsd)
        np.save("./Processed/dataTestingHmsd.npy", dataTestingHmsd)
        np.save("./Processed/labelsTestingHmsd.npy", labelsTestingHmsd)

        dataTrainingHmsadd, labelsTrainingHmsadd, dataTestingHmsadd, labelsTestingHmsadd = HmsaddProcessor()
        np.save("./Processed/dataTrainingHmsadd.npy", dataTrainingHmsadd)
        np.save("./Processed/labelsTrainingHmsadd.npy", labelsTrainingHmsadd)
        np.save("./Processed/dataTestingHmsadd.npy", dataTestingHmsadd)
        np.save("./Processed/labelsTestingHmsadd.npy", labelsTestingHmsadd)

        dataTrainingHMS, labelsTrainingHMS, dataTestingHMS, labelsTestingHMS = HMSProcessor()
        np.save("./Processed/dataTrainingHMS.npy", dataTrainingHMS)
        np.save("./Processed/labelsTrainingHMS.npy", labelsTrainingHMS)
        np.save("./Processed/dataTestingHMS.npy", dataTestingHMS)
        np.save("./Processed/labelsTestingHMS.npy", labelsTestingHMS)

        finalTrainingSet = np.concatenate((dataTrainingMNIST,dataTrainingHMS, dataTrainingHmsd, dataTrainingHmsadd), axis=0)
        np.save("./Processed/finalTrainingSet.npy", finalTrainingSet)
        if path.isfile("./Processed/finalTrainingSet.npy") == True:
            remove("./Processed/dataTrainingMNIST.npy")
            remove("./Processed/dataTrainingHmsd.npy")
            remove("./Processed/dataTrainingHmsadd.npy")
            remove("./Processed/dataTrainingHMS.npy")

        finalTrainingLabels = np.concatenate((labelsTrainingMNIST, labelsTrainingHMS, labelsTrainingHmsd, labelsTrainingHmsadd), axis=0)
        np.save("./Processed/finalTrainingLabels.npy", finalTrainingLabels)
        if path.isfile("./Processed/finalTrainingLabels.npy") == True:
            remove("./Processed/labelsTrainingMNIST.npy")
            remove("./Processed/labelsTrainingHmsd.npy")
            remove("./Processed/labelsTrainingHmsadd.npy")
            remove("./Processed/labelsTrainingHMS.npy")

        finalTestingSet = np.concatenate((dataTestingMNIST, dataTestingHMS, dataTestingHmsd, dataTestingHmsadd), axis=0)
        np.save("./Processed/finalTestingSet.npy", finalTestingSet)
        if path.isfile("./Processed/finalTestingSet.npy") == True:
            remove("./Processed/dataTestingMNIST.npy")
            remove("./Processed/dataTestingHmsd.npy")
            remove("./Processed/dataTestingHmsadd.npy")
            remove("./Processed/dataTestingHMS.npy")

        finalTestingLabels = np.concatenate((labelsTestingMNIST,labelsTestingHMS,labelsTestingHmsd,labelsTestingHmsadd), axis=0)
        np.save("./Processed/finalTestingLabels.npy", finalTestingLabels)
        if path.isfile("./Processed/finalTestingLabels.npy") == True:
            remove("./Processed/labelsTestingMNIST.npy")
            remove("./Processed/labelsTestingHmsd.npy")
            remove("./Processed/labelsTestingHmsadd.npy")
            remove("./Processed/labelsTestingHMS.npy")

    else: #Temporary, do not ship with build
        finalTrainingSet = np.load("./Processed/finalTrainingSet.npy")
        finalTrainingLabels = np.load("./Processed/finalTrainingLabels.npy")
        finalTestingSet = np.load("./Processed/finalTestingSet.npy")
        finalTestingLabels = np.load("./Processed/finalTestingLabels.npy")

        processedTrainingLabels = np.empty((np.shape(finalTrainingLabels)), dtype=np.uint8)
        processedTestingLabels = np.empty((np.shape(finalTestingLabels)), dtype=np.uint8)
        
        counter = 0
        for label in finalTrainingLabels:
            processedTrainingLabels[counter] = tensorDictionary[label]
            counter += 1   
        
        counter = 0
        for label in finalTestingLabels:
            processedTestingLabels[counter] = tensorDictionary[label]
            counter += 1
            
        return finalTrainingSet, processedTrainingLabels, finalTestingSet, processedTestingLabels

trainingSet, trainingLabels, testingSet, testingLabels = MNISTProcessor()

#finalTrainingSet, finalTrainingLabels, finalTestingSet, finalTestingLabels = finalProcessor()
#trainML(finalTrainingSet, finalTrainingLabels, finalTestingSet, finalTestingLabels)
#trainML(trainingSet, trainingLabels.astype(dtype=np.uint8), testingSet, testingLabels.astype(dtype=np.uint8))