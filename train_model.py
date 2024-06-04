import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    images = []
    labels = []
    for label in ['yes', 'no']:
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if file_path.endswith('.png') or file_path.endswith('.jpg'):
                    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    img = img / 255.0
                    images.append(img)
                    labels.append(0 if label == 'no' else 1)
    return np.array(images), np.array(labels)

data_dir = 'c:/projects/braintumor/brain_tumor_dataset'
images, labels = load_data(data_dir)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Model definition
model = Sequential([
    VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save('brain_tumor_detector.h5')
