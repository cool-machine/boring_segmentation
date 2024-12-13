# src/callbacks/reduce_learning_rate_factory.py

from keras.callbacks import ReduceLROnPlateau

def create_reduce_lr(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    cooldown=0,
    verbose=1
):
    """
    Factory function to create a ReduceLROnPlateau callback.

    Args:
        monitor (str): Metric to monitor for reducing learning rate.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        min_lr (float): Lower bound on the learning rate.
        cooldown (int): Number of epochs to wait before resuming normal operation after learning rate is reduced.
        verbose (int): Verbosity mode, 0 = silent, 1 = progress messages.

    Returns:
        keras.callbacks.ReduceLROnPlateau: Configured ReduceLROnPlateau callback.
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        cooldown=cooldown,
        verbose=verbose
    )