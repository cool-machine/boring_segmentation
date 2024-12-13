# src/callbacks/early_stopping_factory.py

from keras.callbacks import EarlyStopping

def create_early_stopping(patience=15, monitor='val_loss', restore_best_weights=True):
    """
    Factory function to create an EarlyStopping callback.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        monitor (str): Metric to monitor for early stopping (e.g., 'val_loss', 'val_accuracy').
        restore_best_weights (bool): Whether to restore the model weights from the epoch 
                                     with the best monitored value.

    Returns:
        keras.callbacks.EarlyStopping: Configured EarlyStopping callback.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
        verbose=1
    )
