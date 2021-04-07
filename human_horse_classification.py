import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

#%% Extract zip file
data_zip  = "zip path"
zip_ref = zipfile.ZipFile(data_zip, 'r')
zip_ref.extractall("path")
zip_ref.close()

horse_train_path = os.path.join("path/Horse_or_Human/horse-or-human/horses")
human_train_path = os.path.join("path/Horse_or_Human/horse-or-human/humans")

#%% Check filenames and find out number of images in that directories.
train_horse_names = os.listdir(horse_train_path)
print(train_horse_names[:10])

train_human_names = os.listdir(human_train_path)
print(train_human_names[:10])

print('number of horse images:', len(os.listdir(horse_train_path)))
print('number of  human images:', len(os.listdir(human_train_path)))

#%% Model building
model = tf.keras.models.Sequential([
    #first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron for binary classification.
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#%% Model summary

model.summary()

#%% Compile model
# Due to it  is a binary classification problem we will train the model with optimizer that is binary_crossentropy
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

#%% Preprocessing
train_datagen = ImageDataGenerator(rescale=1/255) #Normalizing

train_generator = train_datagen.flow_from_directory(
        "path/Horse_or_Human/horse-or-human",  #source directory for training images
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary')

#%% Training
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)

