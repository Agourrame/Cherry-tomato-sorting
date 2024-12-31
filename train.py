import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from utils import load_image 

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Get the total number of training images
train_data_dir = 'data/train/'
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(100, 100),  # Corrected target size
    batch_size=32,
    class_mode='categorical'
)
total_train_images = train_generator.samples 

# Calculate steps_per_epoch considering potential data augmentation
steps_per_epoch = int(np.ceil(total_train_images / train_generator.batch_size))

validation_generator = test_datagen.flow_from_directory(
    'data/test/',
    target_size=(100, 100),  # Corrected target size
    batch_size=32,
    class_mode='categorical'
)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),  # Updated input shape
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes (ripe, semi-ripe, unripe)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (adjust epochs as needed)
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=5,  # Start with a lower number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model.save('app/models/tomato_sorter_model.keras')