# Import necessary libraries
import os
import sys
import keras
import gc
import time

import mlflow
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy


from src.data.processor import load_dataset_segf
from src.utils.metrics import dice_coefficient, iou
from src.utils.optimizers import segf_optimizer
from src.utils.helpers import get_mlflow_uri
from src.utils.loss_funcs import sparse_categorical_crossentropy_loss
from src.architectures.segformer import segformer
from src.training.step import step
from src.callbacks.callbacks import plot_segmentation_results, plot_colored_segmentation

# Import custom callbacks and utilities
from src.callbacks.callbacks import CustomHistory
from src.callbacks.save_best_n_models import maybe_save_best_model

# Dynamically add the `img_segmentation` root directory to `sys.path`
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
print(f"This is the root directory: {root_dir}")


# Optimizer and loss function
model = segformer()
optimizer = segf_optimizer()
loss_fn = sparse_categorical_crossentropy_loss()

# Metrics
mean_train_loss = Mean(name='train_loss')
train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
mean_val_loss = Mean(name='val_loss')
val_accuracy = SparseCategoricalAccuracy(name='val_accuracy')

# Logging
custom_history = CustomHistory()

epochs = 1
best_val_loss = float('inf')
patience = 0
stop_callback = 0
new_lr = optimizer.learning_rate.numpy() 
# Set a global seed for all random ops
tf.random.set_seed(33)

# Enable deterministic ops (if you still want it)
tf.config.experimental.enable_op_determinism()
# Set TensorFlow deterministic behavior
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# Define the main training function
def main():
    """
    Main function to initialize, compile, and train the U-Net model with VGG16 encoder.
    Integrates MLflow for experiment tracking.
    """
    global epochs, best_val_loss, patience, stop_callback, model, loss_fn, optimizer, new_lr

    # Clear Keras backend to ensure a fresh session
    keras.backend.clear_session()
    

    # Configure MLflow tracking URI
    mlflow_tracking_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set the experiment name in MLflow
    experiment_name = "segf_experiment_testing-gpu-02-2025"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="Segformer Training"):


        # Load training and validation datasets
        dataset_train, dataset_val = load_dataset_segf(train_batch_size=32, 
                                                       valid_batch_size=16)

        length_tr = sum([1 for _ in dataset_train])
        length_vl = sum([1 for _ in dataset_val])
        
        print()
        print(f"training dataset contains {length_tr} entries")
        print(f"validation dataset contains {length_vl} entries")
        print()

        # Custom training loop
        custom_history.on_train_begin()

        for epoch in range(epochs):
            start_time = time.time()

            # Reset custom metrics at the start of each epoch
            mean_train_loss.reset_state()
            train_accuracy.reset_state()
            mean_val_loss.reset_state()
            val_accuracy.reset_state()
            
            during_training_steps = 0
            during_training_iou_total = 0
            during_training_dice_total = 0

            train_steps = 0
            train_iou_total = 0
            train_dice_total = 0
            
            val_steps = 0
            val_iou_total = 0
            val_dice_total = 0

            # Training loop
            for images, masks in dataset_train.take(3):
                step(images=images,  
                     masks=masks,
                     model=model,
                     total_loss=mean_train_loss,
                     accuracy=train_accuracy,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     training=True,
                     )

                logits_t = model(images, training=False).logits
                predictions_t = tf.transpose(logits_t, [0, 2, 3, 1])
                masks_t = tf.transpose(masks, [0, 2, 3, 1])
                
                during_training_iou_total += iou(masks_t, predictions_t).numpy()
                during_training_dice_total += dice_coefficient(masks_t, predictions_t).numpy()
                during_training_steps += 1

            during_training_loss = mean_train_loss.result().numpy()
            during_training_accuracy = train_accuracy.result().numpy()
            
            during_training_iou = during_training_iou_total / during_training_steps
            during_training_dice = during_training_dice_total / during_training_steps


            # Compute metrics on training data after training
            for images, masks in dataset_train.take(3):
                step(images=images,  
                     masks=masks,
                     model=model,
                     total_loss=mean_train_loss,
                     accuracy=train_accuracy,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     training=False,
                     )

                masks_tt = tf.transpose(masks, [0, 2, 3, 1])
            
                ## Generate predictions (raw logits) on training data: [B, C(==8), H, W]
                logits_tt = model(images, training=False).logits
                # transpose predictions (raw logits) to channels last: [B, H, W, C(==8)]
                predictions_tt = tf.transpose(logits_tt, [0, 2, 3, 1])

                train_iou_total += iou(masks_tt, predictions_tt).numpy()
                
                train_dice_total += dice_coefficient(masks_tt, predictions_tt).numpy()
                train_steps += 1
            
            epoch_train_loss = mean_train_loss.result().numpy()
            epoch_train_accuracy = train_accuracy.result().numpy()
            
            epoch_train_iou = train_iou_total / train_steps
            epoch_train_dice = train_dice_total / train_steps
            
            # Validation loop
            for val_images, val_masks in dataset_val.take(3):
                
                # shape val_images: [B, C, H, W]
                # shape val_masks: [B, C, H, W]
                step(images=images,  
                     masks=masks,
                     model=model,
                     total_loss=mean_val_loss,
                     accuracy=val_accuracy,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     training=False,
                     )

                # transpose val_masks from channel first: [B, C(==8), H, W] to channel last
                # [B, H, W, C(==8)]
                val_masks_ = tf.transpose(val_masks, [0, 2, 3, 1])
                ## Generate val_predictions (from raw val_logits) on validation data: [B, C(==8), H, W]
                
                val_logits = model(val_images, training=False).logits
                
                # transpose val_logits shape to val_predictions (raw val_logits) 
                # from channels first to channels last: [B, H, W, C(==8)]
                val_predictions = tf.transpose(val_logits, perm=[0, 2, 3, 1])
                
                # val_masks = tf.squeeze(val_masks, axis=-1)
                # accumulate IoU & Dice for validation
                val_iou_total += iou(val_masks_, val_predictions).numpy()
                val_dice_total += dice_coefficient(val_masks_, val_predictions).numpy()
                val_steps += 1

            # # After the entire validation dataset is processed:
            epoch_val_loss = mean_val_loss.result().numpy()
            epoch_val_accuracy = val_accuracy.result().numpy()
            
            epoch_val_iou = val_iou_total / val_steps
            epoch_val_dice = val_dice_total / val_steps

            # Log metrics to the custom history
            custom_history.on_epoch_end(epoch, logs={
                
                'loss': epoch_train_loss,
                'accuracy': epoch_train_accuracy,
                'iou': epoch_train_iou,
                'dice_coefficient': epoch_train_dice,

                'val_loss': epoch_val_loss,
                'val_iou': epoch_val_iou,
                'val_accuracy': epoch_val_accuracy,
                'val_dice_coefficient': epoch_val_dice,                

                'during_training_loss': during_training_loss,
                'during_training_accuracy': during_training_accuracy,
                'during_training_iou': during_training_iou,
                'during_training_dice': during_training_dice,
            })

            # Print current epoch metrics
            print(f'Epoch {epoch + 1}, '
                f'Loss: {epoch_train_loss}, '
                f'Accuracy: {epoch_train_accuracy * 100}, '
                f'IoU: {epoch_train_iou}, '
                f'Dice Coefficient: {epoch_train_dice}, '
        
                f'Val Loss: {epoch_val_loss}, '
                f'Val Accuracy: {epoch_val_accuracy * 100}, '
                f'Val Dice Coefficient: {epoch_val_dice}, '
                f'Val IoU: {epoch_val_iou}, '

                f'During training loss: {during_training_loss}, '
                f'During training accuracy: {during_training_accuracy * 100}, '
                f'During training IoU: {during_training_iou}, '
                f'During training dice: {during_training_dice} ')

            ### 2. Log metrics to MLflow
            mlflow.log_metric('train_loss', epoch_train_loss, step=epoch)
            mlflow.log_metric('train_accuracy', epoch_train_accuracy, step=epoch)
            mlflow.log_metric('iou', epoch_train_iou, step=epoch)        
            mlflow.log_metric('dice_coefficient', epoch_train_dice, step=epoch)

            mlflow.log_metric('val_loss', epoch_val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', epoch_val_accuracy*100, step=epoch)
            mlflow.log_metric('val_iou', epoch_val_iou, step=epoch)
            mlflow.log_metric('val_dice_coefficient', epoch_val_dice, step=epoch)            
            
            mlflow.log_metric('during_training_loss', during_training_loss, step=epoch)
            mlflow.log_metric('during_training_accuracy', during_training_accuracy, step=epoch)
            mlflow.log_metric('during_training_iou', during_training_iou, step=epoch)
            mlflow.log_metric('during_training_dice', during_training_dice, step=epoch)
            
            
            mlflow.log_metric("learning_rate", new_lr)
            mlflow.log_metric("patience", patience)
            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("stop_callback", stop_callback)

            # CALLBACK (1) - reduce learning rate on plateau
            if (patience > 20) and (epoch_val_loss > best_val_loss):
                patience = 0
                new_lr = optimizer.learning_rate.numpy() * 0.5
                maybe_save_best_model(model, epoch_train_iou, epoch)

                if new_lr >= 1e-7:
                    optimizer.learning_rate.assign(new_lr)
                    print(f'Reduced learning rate to {new_lr}')

            # Callback (2) - check for the best model and save at the checkpoint
            if epoch_val_loss < best_val_loss:
                maybe_save_best_model(model, epoch_val_loss, epoch)
                patience = 0
                stop_callback = 0
                best_val_loss = epoch_val_loss
            
            else:
                patience += 1
                stop_callback += 1

            # Callback (3) - plot and save predictions
            if epoch % 10 == 0:
                # for val_images, val_masks in dataset_val.take(1):
                fig0 = plot_segmentation_results(val_images[0],
                                                    val_masks_[0],
                                                    val_predictions[0],
                                                    )
                fig1 = plot_colored_segmentation(val_images[0],
                                                    val_masks_[0],
                                                    val_predictions[0],
                                                    )
                
                mlflow.log_figure(fig0, f"epoch_{epoch}_prediction.png")
                mlflow.log_figure(fig1, f"epoch_{epoch}_prediction_better.png")
                plt.close(fig0)
                plt.close(fig1)
                

            # CALLBACK (4) - check if early stopping condition is met
            if stop_callback > 65:
                print('Early stopping triggered')
                break
            gc.collect()
            elapsed_time = time.time() - start_time
            print(f"\n\nEpoch {epoch + 1}/{epochs} completed in {elapsed_time:.2f} seconds - patience: {patience} - stop_callback: {stop_callback}\n\n")

# Entry point for the script
if __name__ == "__main__":
    # Log script execution start
    print("\n\n")
    print("*" * 60)
    print("Starting Segformer Training with MLflow")
    print("*" * 60)
    print("\n\n")

    # Execute the main function
    main()

    # Log script execution end
    print("\n\n")
    print("*" * 60)
    print("Training Complete")
    print("*" * 60)
    print("\n\n")