# handwritten-latex
AI that converts my terrible handwriting into LaTeX code in order to semi-automatically clean up my notes. This does not take care of formatting. Formatting is left up to the user to deal with by outputting raw LaTeX code for the user to play around with.

I used Tensor Flow (with Python) to train the neural network with data from the [MNIST database](http://yann.lecun.com/exdb/mnist/), the [handwritten math symbols dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols) and my own handwritten data. Credit to those providers for immensely helping creating the training/testing set.

This is my first machine learning project. As such, any kind of suggestion would be welcome. Merge and pull requests are more than welcome, provided that they are well explained.