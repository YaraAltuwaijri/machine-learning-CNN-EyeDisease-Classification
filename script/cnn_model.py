## **CNN Model for Eye Disease Classification**
  
from google.colab import drive
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Mount Google Drive
drive.mount("/content/drive", force_remount=True)

# Path setup
x = "/content/drive/My Drive/RSTAT"
path = Path(x)
print(path)

# Enable Mixed Precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Clear GPU Memory
tf.keras.backend.clear_session()

# Directory paths
train_dir = f"{x}/training"
val_dir = f"{x}/validation"
test_dir = f"{x}/testing"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    zoom_range=0.02,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

batch_size = 16  # Reduced batch size for better generalization

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', shuffle=True)
val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', shuffle=True)
test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical', shuffle=False)

# Compute Class Weights
classes, counts = np.unique(train_generator.classes, return_counts=True)
class_weights = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# Load Pretrained Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:10]:
    layer.trainable = False
for layer in base_model.layers[10:]:
    layer.trainable = True

# Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)  # Stronger L2 Regularization
x = BatchNormalization()(x)
x = Dropout(0.4)(x)  # Increased Dropout to Prevent Overfitting
predictions = Dense(3, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Optimizer with Weight Decay and Gradient Clipping
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=1e-4, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)  # Slower Learning Rate Reduction

# Train Model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Plot Training Results
def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(smooth_curve(history.history['accuracy']), label='Training Accuracy')
plt.plot(smooth_curve(history.history['val_accuracy']), label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(smooth_curve(history.history['loss']), label='Training Loss')
plt.plot(smooth_curve(history.history['val_loss']), label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate Model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))
