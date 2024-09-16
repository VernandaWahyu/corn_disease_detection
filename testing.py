import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Path ke model yang telah disimpan
model_path = 'corn_disease_cnn_model.h5'

# Load model yang sudah ada
model = load_model(model_path)

# Path ke direktori data testing
test_data_dir = 'data/test'

# Tentukan ukuran gambar yang sama dengan yang digunakan saat training
img_height, img_width = 224, 224

# Batch size untuk evaluasi dan prediksi
batch_size = 32

# Augmentasi data untuk generator data testing
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.5,  # Gunakan 50% data sebagai data testing
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Generator untuk data testing
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,  # Jangan acak data testing
    subset='validation'  # Gunakan subset validasi sebagai data testing
)

# Evaluasi model pada data testing
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Prediksi pada data testing
predictions = model.predict(test_generator, verbose=1)

# Konversi prediksi ke label kelas
predicted_classes = np.argmax(predictions, axis=1)

# Dapatkan label sebenarnya
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Hitung dan cetak laporan klasifikasi
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
