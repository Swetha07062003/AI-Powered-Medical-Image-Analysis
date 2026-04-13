import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# -------------------------------
# CREATE FOLDERS
# -------------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# -------------------------------
# PATHS
# -------------------------------
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# -------------------------------
# DATA GENERATORS
# -------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# -------------------------------
# CLASS WEIGHTS (FIX BIAS)
# -------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -------------------------------
# MODEL
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# TRAINING
# -------------------------------
print("\n🚀 Training Started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights
)

# -------------------------------
# SAVE MODEL (UPDATED FORMAT)
# -------------------------------
model.save("model/model.keras")
print("\n✅ Model saved!")

# -------------------------------
# 📊 ACCURACY GRAPH
# -------------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/accuracy.png")
plt.close()

# -------------------------------
# 📉 LOSS GRAPH
# -------------------------------
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/loss.png")
plt.close()

# -------------------------------
# 📊 CONFUSION MATRIX
# -------------------------------
print("\n📊 Generating Confusion Matrix...\n")

test_data.reset()

predictions = model.predict(test_data)
y_pred = (predictions > 0.7).astype(int)   # 🔥 threshold fix
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'Pneumonia'],
    yticklabels=['Normal', 'Pneumonia']
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# -------------------------------
# 📄 CLASSIFICATION REPORT
# -------------------------------
print("\n📄 Classification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=['Normal', 'Pneumonia']
)

print(report)

# Save report
with open("outputs/report.txt", "w") as f:
    f.write(report)

print("\n✅ All outputs saved in 'outputs/' folder")
print("🎯 Project completed successfully!")