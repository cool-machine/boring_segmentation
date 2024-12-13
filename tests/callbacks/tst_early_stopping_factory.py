# import pytest
# from src.callbacks.early_stopping_factory import create_early_stopping
# from keras.callbacks import EarlyStopping

# def test_create_early_stopping():
#     # Create callback
#     callback = create_early_stopping(patience=10, monitor='val_loss')
    
#     # Check that the callback is an instance of EarlyStopping
#     assert isinstance(callback, EarlyStopping)
    
#     # Check that the attributes are set correctly
#     assert callback.patience == 10
#     assert callback.monitor == 'val_loss'
#     assert callback.restore_best_weights is True
