import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Nonaktifkan eager execution
tf.compat.v1.disable_eager_execution()

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
EPOCHS = 10  # Increased epochs
BATCH_SIZE = 32  # Adjusted batch size

# Path to dataset
dataset_path = 'Dataset_CNN'

# Image data generator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Make sure to shuffle the training data
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True  # Shuffle validation data too, although it's not as crucial
)

# Build the AlexNet model
def create_alexnet_model():
    model = Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=256, kernel_size=(5, 5), padding='same'),
        Activation('relu'),
        
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=384, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        Conv2D(filters=384, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(units=4096),
        Activation('relu'),
        Dropout(0.5),
        Dense(units=4096),
        Activation('relu'),
        Dropout(0.5),
        Dense(units=4),
        Activation('softmax')  # Adjusted to 4 classes
    ])
    return model

# Create the model
alexnet_model = create_alexnet_model()

# Compile the model
alexnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = alexnet_model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    use_multiprocessing=False,  # Disable multiprocessing
    callbacks=[reduce_lr, early_stopping]
)

# Save the model
alexnet_model.save('corn_disease_alexnet_model.h5')
