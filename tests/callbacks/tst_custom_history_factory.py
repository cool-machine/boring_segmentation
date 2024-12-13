# import pytest
# from src.callbacks.custom_history_factory import CustomHistory

# def test_custom_history():
#     # Create callback
#     callback = CustomHistory()
    
#     # Simulate training start
#     callback.on_train_begin()
    
#     # Simulate end of epoch
#     logs = {'loss': 0.5, 'accuracy': 0.9, 'val_loss': 0.6, 'val_accuracy': 0.8}
#     callback.on_epoch_end(epoch=0, logs=logs)
    
#     # Check if history is recorded correctly
#     assert callback.history['loss'] == [0.5]
#     assert callback.history['accuracy'] == [0.9]
#     assert callback.history['val_loss'] == [0.6]
#     assert callback.history['val_accuracy'] == [0.8]
