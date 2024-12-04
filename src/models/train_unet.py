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

from data_processing.data_processing import load_dataset_unet
from callbacks.metrics.callbacks import keras_callbacks, PlotResultsCallback
from custom_metrics.custom_metrics import iou, dice_coefficient
from model_definitions.unet import unet_with_vgg16_encoder
from utils.model_architecture_search import unet_optimizer
from logs.mlflow.mlflow_functions import get_mlflow_uri




# define functions
def main():
    # clear Keras backend before compiling and training a new model
    keras.backend.clear_session()    

    # set 'TF_CUDNN_DETERMINISTIC to 1 to avoid approximations (otherwise facing a large errors)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # mlflow set tracking uri
    mlflow_tracking_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # set_tracking_uri()
    
    dataset = load_dataset_unet(batch_size=1)

    # Create an experiment in Azure ML
    experiment_name = "unet_experiment_v1"
    mlflow.set_experiment(experiment_name)
    

    early_stop, reduce_lr, custom_history, top_k_checkpoint = unet_callbacks()
    plot_callback=PlotResultsCallback(validation_data=dataset_val)

    callbacks = [early_stop, reduce_lr, plot_callback, custom_history,top_k_checkpoint]
    metrics = [dice_coefficient, iou, 'accuracy']

    model = unet_with_vgg16_encoder(input_shape, num_classes)
    optimizer = unet_optimizer()


    model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=metrics)

    with mlflow.start_run():
        mlflow.keras.autolog()
        
        history = model.fit(
                    dataset_train.take(1),
                    validation_data=dataset_val.take(1),
                    epochs=2,
                    callbacks=callbacks)




# def evaluate_unet(model, data):
#     all_metrics = []
#     for imgs in data:
#         predictions = model.predict(data['images'])
#         metric1 = compute_metrics1(predictions, data['labels'])
#         metrics2 = compute_metrics2(predictions)
#         all_metrics.append([metric1, metric2])    



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
    