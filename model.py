import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.models import Sequential

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
EPOCHS = 50  # Reduced epochs for demonstration
BATCH_SIZE = 32  # Adjusted batch size

# Path to dataset
dataset_path = 'Dataset_CNN'

# Image data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Function to create the CNN model
def create_model():
    model = Sequential([
        # Feature Learning
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classification
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=4, activation='softmax')  # Adjusted to 4 classes
    ])
    
    return model

# Create the model
cnn_model = create_model()

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1
)

# Save the model
cnn_model.save('corn_disease_cnn_model.h5')
