# handwritten-latex
AI that converts my terrible handwriting into LaTeX code in order to semi-automatically clean up my notes. This does not take care of formatting. Formatting is left up to the user to deal with by outputting raw LaTeX code for the user to rearrange.

I used Tensor Flow (with Python) to train the neural network with data from [MNIST database](http://yann.lecun.com/exdb/mnist/), [handwritten math symbols dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols) and my own handwritten data for missing symbols. Credit to those providers for immensely helping creating the training/testing set. My own images are published in one folder in this repository.

Additional sets used (credit to their respective authors):
- [Handwritten Math Symbols](https://www.kaggle.com/sagyamthapa/handwritten-math-symbols) by sagyamthapa
- [Handwritten math symbol and digit dataset](https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset) by clarencezhao
- [Handwritten Mathematical Expressions](https://www.kaggle.com/rtatman/handwritten-mathematical-expressions) by rtatman

This is my first machine learning project (and first big project involving Python). As such, any kind of suggestion would be welcome. Merge and pull requests are more than welcome, provided that they are well explained. I tried focusing on GUI, machine learning, and general programming structure. Obviously, code is a bit scuffed, but I'll try to improve it as much as possible later down the line.

## Tasks and Planning
I split the project between four different steps:
- Step 1: Train the bot to recognize all signs, digits, letters, etc. This was done by feeding images that are 48x48.
- Step 2: Image processing. Basically, using the cv2 threshold function.
- Step 3: Bounding boxes algorithm to split the digits and letters into multiple images that are then fed to the trained model
- Step 4: Output to the GUI the raw LaTeX code

## Known problems
These are the problems I have yet to address:
- Crappy UI needs redesign
- Magnifying the image does not allow for scrolling or dynamic resizing

## To do
- Retrain the bot with 46+2x46+2 images (only done MNIST so far)
- Integrate training with the UI (yellow button)
- User guide OR labels for the buttons and images