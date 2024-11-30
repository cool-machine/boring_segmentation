# Import libraries
import os
import sys
import keras
import mlflow

# Dynamically add the `img_segmentation` root directory to `sys.path`
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

print(f"this is a root directory: {root_dir}")


from data_processing.image_processing import load_dataset_unet
from callbacks.callbacks import keras_callbacks, PlotResultsCallback
from custom_metrics.custom_metrics import iou, dice_coefficient
from model_definitions.unet import unet_with_vgg16_encoder
from utils.utils import keras_optimizer, get_mlflow_uri



# define functions
def main():

    keras.backend.clear_session()
    
    mlflow_tracking_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Create an experiment in Azure ML
    experiment_name = "unet_experiment_v10"
    mlflow.set_experiment(experiment_name)

    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    dataset_train, dataset_val = load_dataset_unet(batch_size=1)

    early_stop, model_checkpoint, reduce_lr, custom_history, top_k_checkpoint = keras_callbacks()
    plot_callback=PlotResultsCallback(validation_data=dataset_val)

    callbacks = [early_stop, model_checkpoint, reduce_lr, plot_callback, custom_history,top_k_checkpoint]
    metrics = [dice_coefficient, iou, 'accuracy']

    input_shape = (1024, 2048, 3)  # Input shape for Cityscapes dataset
    num_classes = 8  # Number of major groups/classes in mask
    model = unet_with_vgg16_encoder(input_shape, num_classes)
    optimizer = keras_optimizer()
    model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=metrics)

    mlflow.keras.autolog()
    
    with mlflow.start_run():
        
        history = model.fit(
                        dataset_train.take(1),
                        validation_data=dataset_val.take(1),
                        epochs=2,
                        callbacks=callbacks)


# run script
if __name__ == "__main__":
    
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # run main function
    main()

    # add space in logs
    print("*" * 60)
    print("\n\n")
    print("all gone well")