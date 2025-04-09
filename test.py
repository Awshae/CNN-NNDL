import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# GPU Check
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU:", physical_devices)
else:
    print("Running on CPU")

# ✅ Corrected Paths
train_dir = "/Users/asherjarvis/Desktop/CNN/archive_2/train"
test_dir = "/Users/asherjarvis/Desktop/CNN/archive_2/test"

# Hyperparameters
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20

# Data Generators (improved augmentation)
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.3,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

test_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_ds = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Model with MobileNetV2
base_model = MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')
base_model.trainable = True  # ✅ Unfreeze some layers

# ✅ Fine-tune last few layers
for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Compile with AUC metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# Train
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save("mobilenetv2_binary_classifier_final.h5")

# Plot Accuracy and AUC
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluation
y_true = test_ds.classes
y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_ds.class_indices.keys()))
