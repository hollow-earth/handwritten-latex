# handwritten-latex
AI that converts my terrible handwriting into LaTeX code in order to semi-automatically clean up my notes. This does not take care of formatting. Formatting is left up to the user to deal with by outputting raw LaTeX code for the user to rearrange.

I used TensorFlow (with Python) to train the neural network with data my own handwriting as I was not satisfied with what was available out there. My own images are published in [a folder](https://github.com/hollow-earth/handwritten-latex/tree/experimental/Training%20Set/originals) in this repository.

This is my first machine learning project (and first big project involving Python). As such, any kind of suggestion would be welcome. Merge and pull requests are more than welcome, provided that they are well explained. I tried focusing on GUI, machine learning, and general programming structure. Obviously, code is a bit scuffed, but I'll try to improve it as much as possible later down the line.

## Tasks and Planning
I split the project between four different steps:
- [ ] Step 1: Train the bot to recognize all digits, unaccented letters from the English Latin alphabet, common Greek letters, basic mathematical symbols and some advanced ones.
- [ ] Step 2: Image processing. Basically, using the cv2 threshold function
- [ ] Step 3: Bounding boxes algorithm to split the digits and letters into multiple images that are then fed to the trained model
- [ ] Step 4: Output to the GUI the raw LaTeX code

## Potential improvements
- [ ] French Latin alphabet (recognize a few accents, mostly é, è, à, ô, ï, ç)
- [ ] Cyrillic alphabet (Russian)
