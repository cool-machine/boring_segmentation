# Four callbacks for plotting, early stopping, learning rate decay and 
# two checkpointing (one top and one top k models) 

import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# custom made plot callback extending Keras Callback class
class PlotResultsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, plot_interval = 10):
        super(PlotResultsCallback, self).__init__()
        self.validation_data = validation_data
        self.plot_interval = plot_interval

    def on_epoch_end(self, epoch, logs=None):
        # Plot every 'plot_interval' epochs
        if epoch % self.plot_interval == 0:
            # Get a batch of validation data
            val_images, val_masks = next(iter(self.validation_data))

            # Predict the masks
            predictions = self.model(val_images, training=False)
            predicted_masks = K.argmax(predictions, axis=-1)

            # Plot the results
            # for i in range(self.num_samples):
            plt.figure(figsize=(12, 4))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(val_images[0])
            plt.title("Image")
            plt.axis("off")

            # True mask
            plt.subplot(1, 3, 2)
            plt.imshow(val_images[0])
            plt.imshow(tf.squeeze(val_masks[0]),cmap='jet', alpha=0.5)
            plt.title("True Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(val_images[0])
            plt.imshow(predicted_masks[0], cmap='jet', alpha=0.5)
            plt.title("Predicted Mask")
            plt.axis("off")
            plt.show()
            
            # Save the plot to a file
            plot_filename = f'epoch_{epoch}_plot.png'
            plot_path = os.path.join('plots', plot_filename)
            os.makedirs('plots', exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
                    



class TopKModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', mode='auto', top_k=3):
        super(TopKModelCheckpoint, self).__init__()
        self.filepath = filepath  # Filepath template for saving models
        self.monitor = monitor    # Metric to monitor
        self.top_k = top_k        # Number of top models to save
        self.best_models = []     # List to keep track of best models (metric_value, epoch, filepath)

        # Determine whether higher or lower metric is better
        if mode not in ['auto', 'min', 'max']:
            print(f'Mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:  # Auto mode
            if 'acc' in self.monitor or 'accuracy' in self.monitor or 'auc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f'Warning: Metric "{self.monitor}" is not available. Skipping model saving.')
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
            # Check if current model is better than the worst in best_models
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
            print(f'Saved model at {filepath}')

        # Log evolving hyperparameters (e.g., learning rate)
        # Corrected method to get current learning rate
        opt = self.model.optimizer
        lr = opt.learning_rate

        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = lr(opt.iterations)
        else:
            current_lr = lr

        if isinstance(current_lr, tf.Tensor) or isinstance(current_lr, tf.Variable):
            current_lr = current_lr.numpy()
        else:
            current_lr = float(current_lr)


# CustomHistory
class CustomHistory(Callback):

    def on_train_begin(self, logs=None):
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'dice_coefficient': [],
            'val_dice_coefficient': [],
            'iou': [],
            'val_iou': []
        }

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        print(f"Epoch {epoch + 1}: {logs}")



def unet_callbacks(lr_patience=10,
                    es_patience=15):

    # Define the ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # You can also monitor 'val_accuracy'
        factor=0.5,  # Factor by which the learning rate will be reduced
        patience=lr_patience,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbosity mode, 1 = print messages
        mode='auto',  # Mode: 'auto', 'min', or 'max'
        min_delta=0.0001,  # Threshold for measuring the new optimum
        cooldown=0,  # Number of epochs to wait before resuming normal operation after lr has been reduced
        min_lr=0.0000001  # Lower bound on the learning rate
    )

    # Define early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=es_patience,
        verbose=1,
        restore_best_weights=True,
    )

    # # Define the directory for saving checkpoints
    # checkpoint_path = './outputs/checkpoints'
    # os.makedirs(checkpoint_path, exist_ok=True)
    # checkpoint_path = os.path.join(checkpoint_path, 'model-{epoch:03d}-{val_loss:.2f}.keras')
    # model_checkpoint = ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     monitor='val_loss',
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode='auto',
    #     save_freq='epoch',
    #     verbose=1,
    # )

    # Define the directory to save models
    checkpoint_dir = './outputs/top_k_models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir,'model-epoch{epoch:02d}-{val_loss:.4f}.keras')
    # Create an instance of the custom callback
    top_k_checkpoints = TopKModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        mode='min',  # Assuming lower validation loss is better
        top_k=3,      # Save top 3 models
    )
    
    custom_history = CustomHistory()

    return early_stop, reduce_lr, custom_history, top_k_checkpoints
