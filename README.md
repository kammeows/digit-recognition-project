# Project goal

This project aims to recognize handwritten digits in real-time. 
The main goal is to allow a user to write a mathematical expression (like 2 + 3) in any surrounding they wish, and the system will basically read (as in extract) all the operands, understand the operation that is to be performed, and display the correct result.

# Methodology

1. Image Acquisition:

a. Collect digits/symbols written on paper and scan/capture them: I used the addition and subtraction symbols from this kaggle dataset: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
b. Use MNIST dataset for standard digits

2. Image Preprocessing
a. Convert to grayscale
b. Apply thresholding and morphological operations.
c. Detect contours and extract characters via bounding boxes.
d. Resize and normalize each extracted character to 28x28 pixels.
e. Invert if necessary to match MNIST style (white digits on black).

The code for this is written in the file named: preProcess_final.py

3. Model Training
a. Use a CNN architecture trained on the original MNIST dataset first. 
For this, I tried using knn and svm models first but they weren’t able to accurately label the digits. So I experimented with a few different CNN architectures and decided to go with the one having 4 convolutional layers, 3 MaxPoolingLayers and 2 Dense Layers. The chosen model was compiled with the adam optimizer having sparse categorical crossentropy loss. 

This is the model summary:

I trained for approximately 14 epochs with a validation split of 20. 

With this model, I was able to accurately detect digits as shown below:


After finalizing this model, I combined the MNIST dataset with the add and subtract images of the kaggle dataset, and this was the output:

I'm still working on improving the accuracy of the new model.
8. digitRecognition.ipynb:
9. index.html and script.js:
