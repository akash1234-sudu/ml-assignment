from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

with zipfile.ZipFile('/content/drive/MyDrive/Archive.zip', 'r') as z:
    z.extractall('/content/chest_xray')

base = '/content/chest_xray'
for folder in os.listdir(base):
    path = os.path.join(base, folder)
    if os.path.isdir(path) and 'train' in os.listdir(path):
        base = path
        break

train_dir = base + '/train'
test_dir = base + '/test'
val_dir = base + '/val'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode='binary', shuffle=False)
val_data = val_datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode='binary')

print(train_data.class_indices)

imgs, labels = next(train_data)
plt.figure(figsize=(10,4))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(imgs[i])
    plt.title("Pneumonia" if labels[i]==1 else "Normal")
    plt.axis('off')
plt.tight_layout()
plt.show()

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, epochs=15, validation_data=val_data)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.show()

loss, acc = model.evaluate(test_data)
print("Test Accuracy:", round(acc*100, 2), "%")

preds = (model.predict(test_data) > 0.5).astype(int).flatten()
print(classification_report(test_data.classes, preds, target_names=['Normal','Pneumonia']))

cm = confusion_matrix(test_data.classes, preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal','Pneumonia'], yticklabels=['Normal','Pneumonia'])
plt.title('Confusion Matrix')
plt.show()
