"""
Quick prediction script for glaucoma detection
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# Custom layers (needed to load the model)
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(self.filters // self.ratio, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.filters, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((1, 1, self.filters))
        self.multiply = tf.keras.layers.Multiply()

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


class FocalLoss(tf.keras.losses.Loss):
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


class F1Score(tf.keras.metrics.Metric):
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


def find_latest_model():
    """Find the most recent trained model"""
    checkpoint_dir = Path(r'c:\Users\likit\OneDrive\Documents\projects\glucamo\checkpoints')

    if not checkpoint_dir.exists():
        print("No checkpoint folder found!")
        return None

    folders = [f for f in checkpoint_dir.iterdir() if f.is_dir()]
    if not folders:
        print("No model folders found!")
        return None

    latest = max(folders, key=lambda f: f.stat().st_mtime)
    model_path = latest / 'best_model.keras'

    if model_path.exists():
        return model_path
    else:
        print(f"No best_model.keras found in {latest}")
        return None


def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            return None

    print(f"Loading model from: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'SEBlock': SEBlock,
            'FocalLoss': FocalLoss,
            'F1Score': F1Score
        }
    )

    print("[OK] Model loaded successfully!")
    return model


def predict_image(image_path, model, show_gradcam=False):
    """Predict glaucoma for a single image"""

    IMG_SIZE = (224, 224)
    class_labels = ['Normal', 'Glaucoma']

    # Load image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    pred_prob = predictions[0][pred_class]

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {class_labels[pred_class]}")
    print(f"Confidence: {pred_prob:.2%}")
    print(f"\nClass Probabilities:")
    print(f"  Normal:   {predictions[0][0]:.2%}")
    print(f"  Glaucoma: {predictions[0][1]:.2%}")
    print("="*60)

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(img)

    color = 'green' if pred_class == 0 else 'red'
    plt.title(f'Prediction: {class_labels[pred_class]} ({pred_prob:.1%} confidence)',
              fontsize=14, fontweight='bold', color=color)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return pred_class, pred_prob


def batch_predict(folder_path, model, limit=10):
    """Predict for multiple images in a folder"""

    # Get all PNG images
    images = list(Path(folder_path).glob('*.png'))

    if not images:
        print(f"No PNG images found in {folder_path}")
        return

    print(f"\nFound {len(images)} images. Processing first {min(limit, len(images))}...\n")

    results = []
    class_labels = ['Normal', 'Glaucoma']

    for img_path in images[:limit]:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        pred_prob = predictions[0][pred_class]

        results.append({
            'image': img_path.name,
            'prediction': class_labels[pred_class],
            'confidence': pred_prob
        })

        print(f"{img_path.name:<30} → {class_labels[pred_class]:<10} ({pred_prob:.1%})")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("="*60)
    print("GLAUCOMA DETECTION - PREDICTION TOOL")
    print("="*60)

    # Load model
    model = load_model()

    if model is None:
        print("\n[ERROR] Could not load model. Please train the model first.")
        sys.exit(1)

    print("\n" + "="*60)
    print("OPTIONS:")
    print("="*60)
    print("1. Predict single image")
    print("2. Predict batch of images from folder")
    print("3. Test on validation set samples")
    print("="*60)

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        # Single image prediction
        img_path = input("\nEnter image path: ").strip().strip('"')

        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
        else:
            predict_image(img_path, model)

    elif choice == "2":
        # Batch prediction
        folder_path = input("\nEnter folder path: ").strip().strip('"')

        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder not found: {folder_path}")
        else:
            limit = input("How many images to process? (default 10): ").strip()
            limit = int(limit) if limit else 10
            batch_predict(folder_path, model, limit)

    elif choice == "3":
        # Test on validation samples
        val_path = r'c:\Users\likit\OneDrive\Documents\projects\glucamo\Validation-20251107T232720Z-1-001\Validation'

        print("\nTesting on Normal samples:")
        normal_path = os.path.join(val_path, '0')
        if os.path.exists(normal_path):
            batch_predict(normal_path, model, limit=5)

        print("\nTesting on Glaucoma samples:")
        glaucoma_path = os.path.join(val_path, '1')
        if os.path.exists(glaucoma_path):
            batch_predict(glaucoma_path, model, limit=5)

    else:
        print("Invalid choice!")

    print("\n[OK] Done!")
