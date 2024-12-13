import os
import numpy as np
from keras.callbacks import Callback


class TopKModelCheckpoint(Callback):
    """
    A custom Keras callback to save the top K models during training based on a monitored metric.

    Args:
        filepath (str): Template for the path where model files will be saved. Example: 'model-epoch{epoch:02d}-{val_loss:.4f}.h5'.
        monitor (str): Metric to monitor for selecting the top models (e.g., 'val_loss', 'val_accuracy').
        mode (str): One of {'auto', 'min', 'max'}.
                   - 'min': Save models with the lowest metric values.
                   - 'max': Save models with the highest metric values.
                   - 'auto': Automatically detect based on the metric name.
        top_k (int): Number of top models to retain.
    """

    def __init__(self, filepath, monitor='val_loss', mode='auto', top_k=3):
        super(TopKModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.top_k = top_k
        self.best_models = []  # List to track the best models (metric_value, epoch, filepath)

        if mode not in ['auto', 'min', 'max']:
            print(f"Warning: Mode '{mode}' is unknown. Defaulting to 'auto'.")
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            # Automatically determine the operation based on the metric name
            if 'acc' in self.monitor or 'accuracy' in self.monitor or 'auc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f"Warning: Metric '{self.monitor}' not found in logs. Skipping model saving.")
            return

        # Build model info tuple
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        model_info = (current, epoch + 1, filepath)

        # Determine if we should save the model
        save_model = False

        if len(self.best_models) < self.top_k:
            save_model = True
            self.best_models.append(model_info)
        else:
            # Check if the current model is better than the worst in best_models
            if self.monitor_op(current, self.best_models[-1][0]):
                # Remove the worst model
                worst_model = self.best_models.pop()
                worst_filepath = worst_model[2]
                if os.path.exists(worst_filepath):
                    os.remove(worst_filepath)
                save_model = True
                self.best_models.append(model_info)

        if save_model:
            # Sort the best_models list
            self.best_models.sort(key=lambda x: x[0], reverse=self.monitor_op == np.greater)

            # Save the model
            self.model.save(filepath, overwrite=True)
            print(f"Saved model at {filepath}")
