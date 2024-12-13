import pytest
from src.callbacks.reduce_lr_factory import create_reduce_lr
from keras.callbacks import ReduceLROnPlateau

def test_create_reduce_lr():
    # Create callback
    callback = create_reduce_lr(patience=5, factor=0.2, min_lr=1e-6)
    
    # Check that the callback is an instance of ReduceLROnPlateau
    assert isinstance(callback, ReduceLROnPlateau)
    
    # Check that the attributes are set correctly
    assert callback.patience == 5
    assert callback.factor == 0.2
    assert callback.min_lr == 1e-6
