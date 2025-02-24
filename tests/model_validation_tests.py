# tests/model_validation_tests.py
from ../src.data.processor import load_dataset_segf

# tests/model_validation_tests.py

import os
import numpy as np
import tensorflow as tf
import pytest

@pytest.fixture(scope="module")
def load_test_dataset():
    """Fixture to load the entire test dataset for validation."""
    _, _, test_dataset = load_dataset_segf()
    return test_dataset


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Compute the Dice coefficient, a common metric for segmentation performance."""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


@pytest.fixture(scope="module")
def load_keras_model():
    """
    Fixture to load your trained Keras model.
    Adjust the model path to point to your saved model artifact.
    """
    model_path = "path/to/your/saved_model"  # Update this to your actual model path
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def test_segmentation_model(load_keras_model, load_test_dataset):
    """Validate that the segmentation model meets the performance threshold."""
    model = load_keras_model
    _, _, test_data = load_dataset_segf()
    
    images, masks = test_data.take(1)
    # Run inference on the test dataset.
    predictions = model.predict(images)
    
    # Threshold predictions to obtain binary masks.
    predictions_binary = (predictions > 0.5).astype(np.float32)
    
    # Calculate Dice coefficient for each sample.
    dice_scores = [dice_coefficient(y_true, y_pred) for y_true, y_pred in zip(y_test, predictions_binary)]
    avg_dice = np.mean(dice_scores)
    
    expected_dice_threshold = 0.7  # Set an acceptable performance threshold
    assert avg_dice >= expected_dice_threshold, (
        f"Average Dice coefficient {avg_dice:.2f} is below the expected threshold of {expected_dice_threshold:.2f}"
    )
