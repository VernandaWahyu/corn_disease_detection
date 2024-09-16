import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. Muat Model yang Telah Disimpan
model_path = 'corn_disease_cnn_model.h5'
model = load_model(model_path)

# 2. Muat dan Persiapkan Data Testing
# Misalkan data testing berada di folder 'data/test'
test_data_dir = 'data/test'

# Tentukan ukuran gambar yang sama dengan yang digunakan saat training
img_height, img_width = 224, 224  # Contoh ukuran gambar, sesuaikan dengan yang digunakan saat training
batch_size = 32

# Buat ImageDataGenerator untuk data testing dengan 50% data
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,  # Acak data
    subset='validation'  # Gunakan subset validasi sebagai data testing
)

# 3. Lakukan Evaluasi Model pada Data Testing
# Evaluasi model
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# 4. Lakukan Prediksi pada Data Testing
predictions = model.predict(test_generator, verbose=1)

# Konversi prediksi ke label kelas
predicted_classes = np.argmax(predictions, axis=1)

# Dapatkan label sebenarnya
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Hitung dan cetak laporan klasifikasi
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
