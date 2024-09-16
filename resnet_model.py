import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Nonaktifkan eager execution
tf.compat.v1.disable_eager_execution()

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
EPOCHS = 10  # Reduced epochs
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


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(units=512, activation='relu')(x)
predictions = Dense(units=4, activation='softmax')(x)  # Adjusted to 4 classes

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    use_multiprocessing=False  # matikan multiprocessing
)

# Simpan model
model.save('corn_disease_resnet_model.h5')
