import numpy as np
from cv2 import cv2
import os
from random import randint

"""This script is made to take the content of the "Handwritten Math Symbols" (HMS)
and Handwritten math symbols dataset (Hmsd) and splits them 75% training and 25%
testing"""

capitalLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercaseLetters = "abcdefghijklmnopqrstuvwxyz"
path = "./Training Sets/"

def Splitter():
    setsToSplit = ["bruh"]

    for sets in setsToSplit:
        for folder in os.listdir(path + sets +"/training/"):
            if os.path.isdir(path + sets + "./testing/" + folder) == False:
                os.mkdir(path + sets + "./testing/" + folder)
            files = os.listdir(path + sets +"/training/" + folder)
            i = round(0.25 * len(files))
            x = 0
            while i > 0:
                randomNumber = randint(0, len(files)-1-x)
                os.rename(path + sets + "/training/" + folder + "/" + files[randomNumber], path + sets + "/testing/" + folder + "/" + files[randomNumber])
                files.pop(randomNumber)
                i -= 1
                x += 1

"""def Sort():
    for sets in ["Nouveau dossier"]:         #for sets in ["training/", "testing/"]:
        if os.path.isdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/") == False:
            os.mkdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/")
        if os.path.isdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Capital/") == False:
            os.mkdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Capital/")
        if os.path.isdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Lowercase/") == False:
            os.mkdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Lowercase/")

        for folder in os.listdir(path + "Handwritten math symbols dataset/Sorted/"):
            if folder.lower() in lowercaseLetters:
                if os.path.isdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Lowercase/" + folder.lower()) == False:
                    os.mkdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Lowercase/" + folder.lower())
                if os.path.isdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Capital/" + folder.upper()) == False:
                    os.mkdir(path + "Handwritten math symbols dataset/" + sets + "/Sorted/" + "Capital/" + folder.upper())
                files = os.listdir(path + sets +"/training/" + folder)
                print(files)"""

#Sort()
#Splitter()