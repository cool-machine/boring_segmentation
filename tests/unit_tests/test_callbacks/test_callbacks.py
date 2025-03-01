import os
import sys
import shutil
import tempfile
import numpy as np
import pytest
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple

# Ensure the project root is on the module search path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the callbacks module items.
from src.callbacks.callbacks import (
    CustomHistory,
    create_early_stopping,
    PlotResultsCallback,
    plot_segmentation_results,
    plot_colored_segmentation,
    create_reduce_lr,
    maybe_save_best_model,
    TopKModelCheckpoint,
    create_top_k_checkpoint,
    best_models  # Global list used by maybe_save_best_model
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

##########################
# Tests for CustomHistory
##########################
def test_custom_history():
    ch = CustomHistory()
    ch.on_train_begin()
    expected_keys = [
        'loss', 'accuracy', 'val_loss', 'val_accuracy',
        'dice_coefficient', 'val_dice_coefficient',
        'iou', 'val_iou', 'during_training_accuracy',
        'during_training_iou', 'during_training_dice'
    ]
    for key in expected_keys:
        assert key in ch.history
        assert ch.history[key] == []  # Should be empty initially
    
    # Simulate end of epoch with some logs.
    logs = {'loss': 0.5, 'accuracy': 0.8}
    ch.on_epoch_end(0, logs=logs)
    assert ch.history['loss'] == [0.5]
    assert ch.history['accuracy'] == [0.8]

#############################
# Tests for Early Stopping
#############################
def test_create_early_stopping():
    es = create_early_stopping(patience=10, monitor='val_loss', restore_best_weights=True)
    assert isinstance(es, EarlyStopping)
    assert es.patience == 10
    assert es.monitor == 'val_loss'
    assert es.restore_best_weights is True

###########################################
# Tests for Plot Results and Plot Utilities
###########################################
def test_plot_segmentation_results():
    # Create dummy tensors.
    # Dummy image: assume shape (C, H, W) since the function transposes it.
    image = tf.constant(np.random.rand(3, 64, 64), dtype=tf.float32)
    true_mask = tf.constant(np.random.randint(0, 8, (64, 64)), dtype=tf.float32)
    # Dummy predicted mask: shape (H, W, num_classes)
    pred_mask = tf.constant(np.random.rand(64, 64, 8), dtype=tf.float32)
    
    fig = plot_segmentation_results(image, true_mask, pred_mask)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_colored_segmentation():
    image = tf.constant(np.random.rand(3, 64, 64), dtype=tf.float32)
    # Mask with shape (1, H, W)
    mask = tf.constant(np.random.randint(0, 8, (1, 64, 64)), dtype=tf.float32)
    # Predicted mask: shape (H, W, num_classes)
    pred_mask = tf.constant(np.random.rand(64, 64, 8), dtype=tf.float32)
    
    fig = plot_colored_segmentation(image, mask, pred_mask)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

##############################
# Test for Reduce LR Callback
##############################
def test_create_reduce_lr():
    rl = create_reduce_lr(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, cooldown=2, verbose=0)
    assert isinstance(rl, ReduceLROnPlateau)
    assert rl.monitor == 'val_loss'
    assert rl.factor == 0.5
    assert rl.patience == 5

#####################################
# Tests for Model Saving Callbacks
#####################################
# Dummy model for maybe_save_best_model testing.
class DummyModelForSaving:
    def __init__(self):
        self.saved_path = None
    def save_pretrained(self, path):
        self.saved_path = path
        os.makedirs(path, exist_ok=True)

def test_maybe_save_best_model(tmp_path, monkeypatch):
    dummy_model = DummyModelForSaving()
    epoch_val_loss = 0.3
    epoch = 1
    model_name = "dummy_model"
    
    # Clear the global best_models.
    best_models.clear()
    
    # Monkeypatch mlflow.log_artifacts to avoid actual logging.
    monkeypatch.setattr("src.callbacks.callbacks.mlflow.log_artifacts", lambda path, artifact_path: None)
    
    maybe_save_best_model(dummy_model, epoch_val_loss, epoch, model_name)
    
    # Check that best_models now has one entry.
    assert len(best_models) == 1
    entry = best_models[0]
    assert entry["val_loss"] == epoch_val_loss
    assert entry["epoch"] == epoch + 1
    assert os.path.exists(entry["path"])
    
    # Clean up.
    shutil.rmtree(entry["path"])

########################################
# Tests for Top-K Model Checkpoint Callback
########################################

class DummyTopKModelCheckpoint(TopKModelCheckpoint):
    def set_model(self, model):
        # Instead of trying to set the built‐in "model" attribute,
        # store it in a custom private attribute.
        self._dummy_model = model

    @property
    def model(self):
        return self._dummy_model

class DummyModelForCheckpoint:
    def save(self, filepath, overwrite=True):
        # Simulate saving by creating an empty file.
        with open(filepath, "w") as f:
            f.write("dummy model content")


def test_topk_model_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    filepath_template = str(checkpoint_dir / "model-epoch{epoch:02d}-{val_loss:.4f}.keras")
    # Use the dummy subclass that allows setting the model
    tk = DummyTopKModelCheckpoint(filepath=filepath_template, monitor='val_loss', mode='min', top_k=2)
    
    dummy_model = DummyModelForCheckpoint()
    tk.set_model(dummy_model)  # This now sets our custom _dummy_model attribute.
    
    # Simulate several epochs.
    logs_epoch1 = {'val_loss': 0.5}
    tk.on_epoch_end(0, logs=logs_epoch1)
    logs_epoch2 = {'val_loss': 0.4}
    tk.on_epoch_end(1, logs=logs_epoch2)
    logs_epoch3 = {'val_loss': 0.6}
    tk.on_epoch_end(2, logs=logs_epoch3)
    
    # Check that at most 2 models are kept.
    assert len(tk.best_models) <= 2
    for (_, _, path) in tk.best_models:
        assert os.path.exists(path)
        os.remove(path)  # Clean up saved file.


# def test_topk_model_checkpoint(tmp_path, monkeypatch):
#     checkpoint_dir = tmp_path / "checkpoints"
#     checkpoint_dir.mkdir()
#     filepath_template = str(checkpoint_dir / "model-epoch{epoch:02d}-{val_loss:.4f}.keras")
#     tk = TopKModelCheckpoint(filepath=filepath_template, monitor='val_loss', mode='min', top_k=2)
    
#     dummy_model = DummyModelForCheckpoint()
#     object.__setattr__(tk, "model", dummy_model)   # Set the model for the callback.
    
#     # Simulate several epochs.
#     logs_epoch1 = {'val_loss': 0.5}
#     tk.on_epoch_end(0, logs=logs_epoch1)
#     logs_epoch2 = {'val_loss': 0.4}
#     tk.on_epoch_end(1, logs=logs_epoch2)
#     logs_epoch3 = {'val_loss': 0.6}
#     tk.on_epoch_end(2, logs=logs_epoch3)
    
#     # Check that at most 2 models are kept.
#     assert len(tk.best_models) <= 2
#     for (_, _, path) in tk.best_models:
#         assert os.path.exists(path)
#         os.remove(path)  # Clean up saved file.

def test_create_top_k_checkpoint(tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tk_checkpoint = create_top_k_checkpoint(checkpoint_dir=checkpoint_dir, top_k=2, monitor='val_loss', mode='min')
    from src.callbacks.callbacks import TopKModelCheckpoint
    assert isinstance(tk_checkpoint, TopKModelCheckpoint)
