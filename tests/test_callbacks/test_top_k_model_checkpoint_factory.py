
# import pytest
# from src.callbacks.reduce_lr_factory import create_reduce_lr
# from keras.callbacks import ReduceLROnPlateau

# def test_create_reduce_lr():
#     # Create callback
#     callback = create_reduce_lr(patience=5, factor=0.2, min_lr=1e-6)
    
#     # Check that the callback is an instance of ReduceLROnPlateau
#     assert isinstance(callback, ReduceLROnPlateau)
    
#     # Check that the attributes are set correctly
#     assert callback.patience == 5
#     assert callback.factor == 0.2
#     assert callback.min_lr == 1e-6

import os
import pytest
from unittest.mock import MagicMock
from src.callbacks.top_k_checkpoint_factory import create_top_k_checkpoint
from src.callbacks.top_k_model_checkpoint import TopKModelCheckpoint


def test_create_top_k_checkpoint(tmpdir):
    # Create a temporary directory for checkpoints
    checkpoint_dir = tmpdir.mkdir("test_checkpoints")
    
    # Create callback
    callback = create_top_k_checkpoint(checkpoint_dir=str(checkpoint_dir), top_k=3)
    
    # Mock the model and set it using the `set_model` method
    mock_model = MagicMock()
    callback.set_model(mock_model)
    
    # Simulate saving a model
    callback.on_epoch_end(epoch=1, logs={'val_loss': 0.5})
    
    # Check if the file exists
    expected_filepath = os.path.join(str(checkpoint_dir), 'model-epoch02-0.5000.h5')
    assert os.path.exists(expected_filepath), f"Expected file {expected_filepath} not found!"
    
    # Ensure the directory contains the expected file
    assert len(list(checkpoint_dir.listdir())) == 1




# def test_create_top_k_checkpoint(tmpdir):
#     # Create a temporary directory for checkpoints
#     checkpoint_dir = tmpdir.mkdir("test_checkpoints")
    
#     # Create callback
#     callback = create_top_k_checkpoint(checkpoint_dir=str(checkpoint_dir), top_k=3)
    
#     # Check that the callback is an instance of TopKModelCheckpoint
#     assert isinstance(callback, TopKModelCheckpoint)
    
#     # Mock the model and set it using the `set_model` method
#     mock_model = MagicMock()
#     callback.set_model(mock_model)
    
#     # Simulate saving a model and check the directory contents
#     callback.on_epoch_end(epoch=1, logs={'val_loss': 0.5})
    
#     # Ensure a checkpoint file is created
#     assert len(list(checkpoint_dir.listdir())) == 1
