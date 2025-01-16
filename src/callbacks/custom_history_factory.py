# src/callbacks/custom_history_factory.py

from keras.callbacks import Callback

class CustomHistory(Callback):
    """
    Custom callback to track additional metrics during training.

    This callback maintains a history dictionary to record the values of
    specified metrics at the end of each epoch.

    Attributes:
        history (dict): Dictionary containing recorded metrics.

    Methods:
        on_train_begin: Initializes the history dictionary at the start of training.
        on_epoch_end: Updates the history dictionary at the end of each epoch.
    """

    def on_train_begin(self, logs=None):
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'dice_coefficient': [],
            'val_dice_coefficient': [],
            'iou': [],
            'val_iou': [],
            'during_training_accuracy':[],
            'during_training_iou':[],
            'during_training_dice':[],
        }

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        print(f"Epoch {epoch + 1}: {logs}")