# Chest X-Ray Classification - CNN

## Dataset
5863 chest x-ray images divided into Normal and Pneumonia.
Folders: train, test, val

## What I did
I used a CNN model to classify x-ray images.
First loaded the images using ImageDataGenerator with some augmentation.
Then built the model with 3 conv layers and trained it.

## Model
- Conv2D(32) + MaxPool
- Conv2D(64) + MaxPool
- Conv2D(128) + MaxPool
- Dense(128) + Dropout
- Dense(1, sigmoid)

## Training
- Optimizer: adam
- Loss: binary crossentropy
- Epochs: 15

## Result
Checked accuracy and confusion matrix on test set.
Model can detect pneumonia from x-ray images.

## How to run
Upload Archive.zip to Google Drive and run in Colab.
