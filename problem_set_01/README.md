# Chest X-Ray Classification - CNN

## Dataset
5863 chest x-ray images divided into Normal and Pneumonia.  
Folders: train, test, val  

## What I did
I used a CNN model to classify x-ray images.  
First loaded the images using ImageDataGenerator with augmentation.  
Then built the CNN model and trained it.  

## Model
- Conv2D(32) + BatchNormalization + MaxPool  
- Conv2D(64) + BatchNormalization + MaxPool  
- Conv2D(128) + BatchNormalization + MaxPool  
- Conv2D(256) + BatchNormalization + MaxPool  
- Dense(256) + Dropout  
- Dense(1, sigmoid)  

## Training
- Optimizer: Adam  
- Loss: binary crossentropy  
- Epochs: 20  
## Result
Checked accuracy and confusion matrix on test set.
Model can detect pneumonia from x-ray images.

## How to run
Upload Archive.zip to Google Drive and run in Colab.
