# Getting Started - Local Training Guide

Complete guide to train the enhanced glaucoma detection model on your local Windows machine.

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- GPU (optional, but recommended for faster training)
- ~2GB free disk space for models and results

## Step-by-Step Setup

### Step 1: Install Python Packages

Open **PowerShell** or **Command Prompt** and run:

```bash
# Navigate to project folder
cd c:\Users\likit\OneDrive\Documents\projects\glucamo

# Install all required packages
pip install tensorflow tensorflow-addons scikit-learn matplotlib seaborn opencv-python pandas jupyter
```

**Note:** This may take 5-10 minutes depending on your internet speed.

### Step 2: Verify Installation

```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed'); print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"
```

Expected output:
```
TensorFlow 2.x.x installed
GPU Available: [list of GPUs or empty list]
```

### Step 3: Verify Data Folders

```bash
# Check if data folders exist
dir Train-20251107T233046Z-1-001\Train
dir Validation-20251107T232720Z-1-001\Validation
dir Test-20251108T015821Z-1-001\Test
```

You should see folders named `0` and `1` containing images.

## Training Options

### Option A: Using Jupyter Notebook (Recommended)

#### 1. Start Jupyter Notebook

```bash
jupyter notebook Enhanced_Glaucoma_Detection_Local.ipynb
```

Your browser will open automatically.

#### 2. Run Training

In Jupyter:
- Click **Cell → Run All**
- Or press **Shift+Enter** on each cell

#### 3. Monitor Progress

The notebook shows real-time progress:
```
Epoch 1/50
166/166 [==============================] - 45s 271ms/step - loss: 0.4210 - accuracy: 0.7820
```

### Option B: Using Python Script Monitor

If you want a cleaner monitoring interface:

#### 1. Start Training in Jupyter (as above)

#### 2. Open a NEW PowerShell/Terminal and run:

```bash
cd c:\Users\likit\OneDrive\Documents\projects\glucamo
python monitor_training.py
```

This will show:
```
================================================================================
TRAINING MONITOR - glaucoma_custom_cnn_20241107_153045
================================================================================

Epoch: 15/50

Latest Epoch (15):
  Loss:          0.1982
  Accuracy:      0.9210 (92.10%)
  Val Loss:      0.2145
  Val Accuracy:  0.9015 (90.15%)
  Val AUC:       0.9523
  Val Recall:    0.8567
  Val Precision: 0.8923

Best Results So Far:
  Best Val Acc:  0.9015 (Epoch 15)
  Best Val AUC:  0.9523 (Epoch 15)

Trend: 📈 Val Accuracy improved
================================================================================
Last updated: 2024-11-07 15:45:23
Refreshing every 5 seconds... (Ctrl+C to stop)
```

### Option C: Using TensorBoard (Advanced)

If you installed TensorBoard:

#### 1. Start Training (Jupyter)

#### 2. In a NEW terminal:

```bash
# Wait for training to create the checkpoint folder first
# Then run (replace the timestamp with actual folder name):

tensorboard --logdir="c:\Users\likit\OneDrive\Documents\projects\glucamo\checkpoints"
```

#### 3. Open browser to:

```
http://localhost:6006
```

## Training Time Estimates

| Hardware | Estimated Time (50 epochs) |
|----------|---------------------------|
| GPU (NVIDIA) | 20-30 minutes |
| CPU only | 4-6 hours |

**Recommendation:** If you only have CPU, reduce epochs to 20-30 in the notebook:
```python
EPOCHS = 20  # Instead of 50
```

## What Happens During Training

### Automatic Actions:

1. **Data Loading** - Loads images from your local folders
2. **Class Balancing** - Computes weights for imbalanced data
3. **Model Building** - Creates custom CNN architecture
4. **Training** - Trains for 50 epochs with early stopping
5. **Evaluation** - Generates metrics and visualizations
6. **Saving** - Exports models in multiple formats

### Files Created:

All outputs saved to: `checkpoints\glaucoma_custom_cnn_<timestamp>\`

```
checkpoints\
└── glaucoma_custom_cnn_20241107_153045\
    ├── best_model.keras           ← Use this for predictions
    ├── final_model.keras
    ├── final_model.h5
    ├── final_model.tflite         ← For mobile apps
    ├── saved_model\                ← For cloud deployment
    ├── logs\                       ← TensorBoard logs
    ├── training_log.csv           ← All metrics per epoch
    ├── training_history.png       ← Training curves
    ├── confusion_matrix.png       ← Performance visualization
    ├── roc_curve.png              ← ROC curve
    ├── precision_recall_curve.png
    ├── classification_report.txt
    └── training_summary.txt
```

## After Training is Complete

### View Results

The notebook automatically opens the results folder. You'll find:

#### 1. Model Files
- `best_model.keras` - Best model (use this!)
- `final_model.tflite` - For mobile deployment

#### 2. Visualizations
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Classification performance
- `roc_curve.png` - ROC curve with optimal threshold

#### 3. Reports
- `classification_report.txt` - Detailed metrics
- `training_summary.txt` - Overall summary
- `training_log.csv` - Epoch-by-epoch data

### Make Predictions

In Jupyter notebook, use:

```python
# Predict a single image with Grad-CAM visualization
predict_single_image(
    r'c:\Users\likit\OneDrive\Documents\projects\glucamo\Test-20251108T015821Z-1-001\Test\1\image.png',
    model,
    show_gradcam=True
)
```

## Troubleshooting

### Problem: "Out of Memory" Error

**Solution:**
```python
# In the notebook, reduce batch size:
BATCH_SIZE = 16  # or even 8
```

### Problem: Training is Very Slow

**Solutions:**
1. Check if GPU is being used:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. Reduce epochs:
   ```python
   EPOCHS = 20  # Instead of 50
   ```

3. Enable mixed precision (add to notebook):
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

### Problem: Can't Find Training Data

**Solution:**
```python
# Verify paths in the notebook configuration cell:
print(f"Training path exists: {os.path.exists(TRAINING_PATH)}")
print(f"Validation path exists: {os.path.exists(VALIDATION_PATH)}")
```

Check folder structure:
```
glucamo\
├── Train-20251107T233046Z-1-001\
│   └── Train\
│       ├── 0\  ← Normal images
│       └── 1\  ← Glaucoma images
├── Validation-20251107T232720Z-1-001\
│   └── Validation\
│       ├── 0\
│       └── 1\
└── Test-20251108T015821Z-1-001\
    └── Test\
        ├── 0\
        └── 1\
```

### Problem: Import Errors

**Solution:**
```bash
# Reinstall packages
pip uninstall tensorflow tensorflow-addons
pip install tensorflow tensorflow-addons
```

## Performance Tips

### Speed Up Training:

1. **Use GPU** - 10-15× faster than CPU
2. **Increase Batch Size** - If you have enough RAM:
   ```python
   BATCH_SIZE = 64  # Instead of 32
   ```

3. **Reduce Image Size** - If accuracy is acceptable:
   ```python
   IMG_SIZE = (128, 128)  # Instead of (224, 224)
   ```

### Improve Accuracy:

1. **More Epochs** - Let it train longer:
   ```python
   EPOCHS = 100
   ```

2. **Adjust Learning Rate**:
   ```python
   LEARNING_RATE = 0.0001  # Lower for more stable training
   ```

3. **More Augmentation**:
   ```python
   train_datagen = ImageDataGenerator(
       rotation_range=30,  # More rotation
       zoom_range=0.3,     # More zoom
       # ... etc
   )
   ```

## Next Steps After Training

1. **Evaluate on Test Set** - Use the test data to verify performance
2. **Try Different Hyperparameters** - Experiment with learning rate, epochs
3. **Ensemble Models** - Train multiple models and average predictions
4. **Deploy** - Use the .tflite model for mobile apps

## Need Help?

Check the documentation files:
- `README.md` - Complete guide
- `IMPROVEMENTS.md` - Technical details
- `MODEL_COMPARISON.md` - Original vs Enhanced comparison
- `QUICK_REFERENCE.md` - Quick reference card

## Summary Commands

```bash
# 1. Install packages
pip install tensorflow tensorflow-addons scikit-learn matplotlib seaborn opencv-python pandas jupyter

# 2. Start training
jupyter notebook Enhanced_Glaucoma_Detection_Local.ipynb

# 3. Monitor (optional, in new terminal)
python monitor_training.py

# 4. View results
explorer checkpoints\glaucoma_custom_cnn_<timestamp>
```

---

**Ready to start?** Just run:

```bash
jupyter notebook Enhanced_Glaucoma_Detection_Local.ipynb
```

Then click **Cell → Run All** and wait for training to complete!
