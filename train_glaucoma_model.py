"""
Enhanced Glaucoma Detection - Python Script Version
No tensorflow-addons required!
Compatible with Python 3.13
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import cv2
import os
from datetime import datetime
import glob

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {__import__('sys').version}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = r'c:\Users\likit\OneDrive\Documents\projects\glucamo'

TRAINING_PATH = os.path.join(BASE_PATH, 'Train-20251107T233046Z-1-001', 'Train')
VALIDATION_PATH = os.path.join(BASE_PATH, 'Validation-20251107T232720Z-1-001', 'Validation')
TEST_PATH = os.path.join(BASE_PATH, 'Test-20251108T015821Z-1-001', 'Test')

# Verify paths exist
print("\nChecking data paths...")
print(f"Training path exists: {os.path.exists(TRAINING_PATH)}")
print(f"Validation path exists: {os.path.exists(VALIDATION_PATH)}")
print(f"Test path exists: {os.path.exists(TEST_PATH)}")

# Model configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 2
LEARNING_RATE = 0.001

# Model save path
MODEL_NAME = f'glaucoma_custom_cnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'checkpoints', MODEL_NAME)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print(f"\nModels will be saved to: {CHECKPOINT_PATH}")

# ============================================================================
# CUSTOM LAYERS
# ============================================================================

class SEBlock(layers.Layer):
    """Squeeze-and-Excitation Block for Attention"""
    def __init__(self, filters, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.filters // self.ratio, activation='relu')
        self.dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.filters))
        self.multiply = layers.Multiply()

    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return self.multiply([inputs, se])

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "ratio": self.ratio})
        return config


def residual_block_with_se(x, filters, kernel_size=3, stride=1, use_se=True):
    """Residual block with optional Squeeze-and-Excitation"""
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    if use_se:
        x = SEBlock(filters)(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def inception_block(x, filters):
    """Multi-scale feature extraction"""
    branch1 = layers.Conv2D(filters, 1, padding='same', activation='relu',
                           kernel_initializer='he_normal')(x)

    branch2 = layers.Conv2D(filters, 1, padding='same', activation='relu',
                           kernel_initializer='he_normal')(x)
    branch2 = layers.Conv2D(filters, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal')(branch2)

    branch3 = layers.Conv2D(filters, 1, padding='same', activation='relu',
                           kernel_initializer='he_normal')(x)
    branch3 = layers.Conv2D(filters, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal')(branch3)
    branch3 = layers.Conv2D(filters, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal')(branch3)

    branch4 = layers.MaxPooling2D(3, strides=1, padding='same')(x)
    branch4 = layers.Conv2D(filters, 1, padding='same', activation='relu',
                           kernel_initializer='he_normal')(branch4)

    output = layers.Concatenate()([branch1, branch2, branch3, branch4])
    output = layers.BatchNormalization()(output)

    return output


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow(1 - y_pred, self.gamma)

        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ============================================================================
# CUSTOM F1 METRIC
# ============================================================================

class F1Score(tf.keras.metrics.Metric):
    """F1 Score metric"""
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


# ============================================================================
# BUILD MODEL
# ============================================================================

def build_custom_medical_cnn(input_shape=(224, 224, 3), num_classes=2):
    """Custom CNN architecture for medical imaging"""
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                     kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Stage 1: Residual blocks with SE
    x = residual_block_with_se(x, 64, use_se=True)
    x = residual_block_with_se(x, 64, use_se=True)

    # Stage 2: Multi-scale inception block + residual
    x = inception_block(x, 32)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = residual_block_with_se(x, 128, use_se=True)
    x = residual_block_with_se(x, 128, use_se=True)

    # Stage 3: Deeper features
    x = residual_block_with_se(x, 256, stride=2, use_se=True)
    x = residual_block_with_se(x, 256, use_se=True)
    x = residual_block_with_se(x, 256, use_se=True)

    # Stage 4: High-level features
    x = inception_block(x, 64)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = residual_block_with_se(x, 512, use_se=True)
    x = residual_block_with_se(x, 512, use_se=True)

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax',
                          kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=inputs, outputs=outputs, name='MedicalImageCNN')

    return model


print("\n" + "="*80)
print("BUILDING MODEL")
print("="*80)

model = build_custom_medical_cnn(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
model.summary()

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='reflect'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print(f"\nTraining samples: {train_generator.n}")
print(f"Validation samples: {validation_generator.n}")
print(f"Class indices: {train_generator.class_indices}")

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

class_counts = np.bincount(train_generator.classes)
print(f"\nClass distribution: {class_counts}")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ============================================================================
# COMPILE MODEL
# ============================================================================

print("\n" + "="*80)
print("COMPILING MODEL")
print("="*80)

USE_FOCAL_LOSS = True

if USE_FOCAL_LOSS:
    loss = FocalLoss(gamma=2.0, alpha=0.25)
    print("Using Focal Loss")
else:
    loss = 'categorical_crossentropy'
    print("Using Categorical Crossentropy")

# Use built-in AdamW (available in TensorFlow 2.11+)
try:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4
    )
    print("Using AdamW optimizer with weight decay")
except AttributeError:
    # Fallback to regular Adam if AdamW not available
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    print("Using Adam optimizer (AdamW not available)")

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        F1Score(name='f1_score')
    ]
)

print("[OK] Model compiled successfully!")

# ============================================================================
# CALLBACKS
# ============================================================================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_PATH, 'best_model.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    TensorBoard(
        log_dir=os.path.join(CHECKPOINT_PATH, 'logs'),
        histogram_freq=1,
        write_graph=True
    ),

    CSVLogger(
        os.path.join(CHECKPOINT_PATH, 'training_log.csv'),
        append=False
    )
]

print("[OK] Callbacks configured!")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"\nEpochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"\nTo monitor training, run in another terminal:")
print(f'python monitor_training.py')
print("="*80 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weight_dict if not USE_FOCAL_LOSS else None,
    callbacks=callbacks,
    verbose=1
)

print("\n[OK] Training completed!")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

model.save(os.path.join(CHECKPOINT_PATH, 'final_model.keras'))
print("[OK] Keras format (.keras)")

model.save(os.path.join(CHECKPOINT_PATH, 'final_model.h5'))
print("[OK] H5 format (.h5)")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(os.path.join(CHECKPOINT_PATH, 'final_model.tflite'), 'wb') as f:
    f.write(tflite_model)
print("[OK] TFLite format (.tflite)")

model.save(os.path.join(CHECKPOINT_PATH, 'saved_model'), save_format='tf')
print("[OK] SavedModel format")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("EVALUATING MODEL")
print("="*80)

validation_generator.reset()
y_true = validation_generator.classes
y_pred_probs = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = ['Normal', 'Glaucoma']

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
roc_auc = roc_auc_score(y_true, y_pred_probs[:, 1])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"\n{'Accuracy':.<30} {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'Precision':.<30} {precision:.4f}")
print(f"{'Recall (Sensitivity)':.<30} {recall:.4f}")
print(f"{'Specificity':.<30} {specificity:.4f}")
print(f"{'F1 Score':.<30} {f1:.4f}")
print(f"{'ROC-AUC':.<30} {roc_auc:.4f}")

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

with open(os.path.join(CHECKPOINT_PATH, 'classification_report.txt'), 'w') as f:
    f.write(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# Summary
with open(os.path.join(CHECKPOINT_PATH, 'training_summary.txt'), 'w') as f:
    f.write("TRAINING SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: Custom Medical Imaging CNN\n")
    f.write(f"Parameters: {model.count_params():,}\n")
    f.write(f"Training Samples: {train_generator.n}\n")
    f.write(f"Validation Samples: {validation_generator.n}\n")
    f.write(f"Epochs: {len(history.history['loss'])}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Training history
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Training History', fontsize=16, fontweight='bold')

metrics = [
    ('loss', 'Loss'),
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('auc', 'AUC'),
    ('f1_score', 'F1 Score')
]

for idx, (metric, title) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    if metric in history.history:
        ax.plot(history.history[metric], label=f'Train {title}', linewidth=2)
        ax.plot(history.history[f'val_{metric}'], label=f'Val {title}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{title} Over Epochs', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_PATH, 'training_history.png'), dpi=300, bbox_inches='tight')
print("[OK] Training history plot saved")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, normalize='true')
cm_percentage = cm * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_PATH, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("[OK] Confusion matrix saved")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
plt.scatter(fpr[ix], tpr[ix], marker='o', color='red', s=200,
            label=f'Optimal Threshold = {best_thresh:.3f}')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_PATH, 'roc_curve.png'), dpi=300, bbox_inches='tight')
print("[OK] ROC curve saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {CHECKPOINT_PATH}")
print("\nGenerated files:")
print("  - best_model.keras")
print("  - final_model.keras")
print("  - final_model.h5")
print("  - final_model.tflite")
print("  - saved_model/")
print("  - training_log.csv")
print("  - training_history.png")
print("  - confusion_matrix.png")
print("  - roc_curve.png")
print("  - classification_report.txt")
print("  - training_summary.txt")
print("="*80)

# Open results folder
import subprocess
subprocess.Popen(f'explorer "{CHECKPOINT_PATH}"')
print(f"\n[OK] Opened results folder in Windows Explorer")
