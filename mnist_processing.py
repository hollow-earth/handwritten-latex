import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from mnist import MNIST

path = "C:/Users/Sam/Desktop/Training Sets/MNIST/"
mndata = MNIST(path)

imagesTraining, labelsTraining = mndata.load_training()
imagesTesting, labelsTesting = mndata.load_testing()

for i in range(0, 3):
    image = np.array(imagesTraining[i]).reshape(28,28).astype("float32")
    image /= 255
    io.imsave(path + str(i) + ".jpg", image)