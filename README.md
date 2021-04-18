# handwritten-latex
AI that converts my terrible handwriting into LaTeX code in order to semi-automatically clean up my notes. This does not take care of formatting. Formatting is left up to the user to deal with by outputting raw LaTeX code for the user to play around with.

I used Tensor Flow (with Python) to train the neural network with data from the [MNIST database](http://yann.lecun.com/exdb/mnist/), the [handwritten math symbols dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols) and my own handwritten data for missing symbols. Credit to those providers for immensely helping creating the training/testing set. My own images are published in one folder in this repo.

Additional sets used (credit to their respective authors):
- [Handwritten Math Symbols](https://www.kaggle.com/sagyamthapa/handwritten-math-symbols)
- [Handwritten math symbol and digit dataset](https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset)
- [Handwritten Mathematical Expressions](https://www.kaggle.com/rtatman/handwritten-mathematical-expressions)

This is my first machine learning project. As such, any kind of suggestion would be welcome. Merge and pull requests are more than welcome, provided that they are well explained.