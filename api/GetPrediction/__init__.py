import logging
import azure.functions as func
import json
import base64
import os
import sys
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Add the api directory to the path so we can import from src
api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(api_dir)

# Import Azure utilities
from src.utils.azure_utils import download_blob_to_memory, get_blob_service_client

def image_to_base64(image_array):
    """Convert a numpy array image to base64 string using matplotlib"""
    # Create a new figure with tight layout
    fig = plt.figure(figsize=(10, 10), frameon=False, dpi=100)
    canvas = FigureCanvas(fig)
    
    # If it's a mask with values 0, 1, 2, use a colormap
    if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[2] == 1):
        plt.imshow(image_array, cmap='viridis', vmin=0, vmax=2)
    else:
        plt.imshow(image_array)
    
    plt.axis('off')  # Turn off axis
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    
    # Encode as base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

def create_simple_segmentation(image_array):
    """
    Create a simple segmentation mask for demonstration purposes.
    In a real application, this would use a trained model.
    
    Args:
        image_array: Numpy array representing the image
        
    Returns:
        Numpy array: A simple segmentation mask
    """
    # Create a simple mask based on color thresholds (this is just for demonstration)
    # In a real application, you would use your actual segmentation model here
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Simple thresholding on the green channel for demonstration
    if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
        # Threshold on green channel
        green_channel = image_array[:, :, 1]
        mask[green_channel > 100] = 1  # Class 1 for green-ish areas
        mask[green_channel > 180] = 2  # Class 2 for very green areas
    
    return mask


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
    # Log environment variables for debugging
    logging.info("Function environment:")
    for key, value in os.environ.items():
        if "KEY" not in key and "SECRET" not in key and "PASSWORD" not in key:
            logging.info(f"{key}: {value}")
    
    try:
        # Parse request body
        req_body = req.get_json()
        image_path = req_body.get('image_path')
        image_container = req_body.get('image_container', os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1"))
        
        logging.info(f"Processing image: {image_path} from container: {image_container}")
        
        # Validate input
        if not image_path:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "No image_path provided in request"}),
                mimetype="application/json",
                status_code=400
            )
        
        try:
            # Try to get the blob service client
            blob_service_client = get_blob_service_client()
            logging.info(f"Successfully created blob service client")
            
            # Download the image from Azure Blob Storage
            image_bytes = download_blob_to_memory(image_path, container_name=image_container)
            logging.info(f"Successfully downloaded image: {image_path}")
            
            # Open the image with matplotlib
            image_buffer = io.BytesIO(image_bytes)
            image_array = mpimg.imread(image_buffer)
            logging.info(f"Image shape: {image_array.shape}")
            
            # Create a simple segmentation mask (in a real app, this would use your model)
            mask = create_simple_segmentation(image_array)
            logging.info(f"Created segmentation mask")
            
            # For demonstration, use the same mask for ground truth and prediction
            # In a real application, you would load the ground truth from storage
            # and generate the prediction using your model
            ground_truth = mask
            prediction = mask
            
            # Convert images to base64
            original_base64 = image_to_base64(image_array)
            ground_truth_base64 = image_to_base64(ground_truth)
            prediction_base64 = image_to_base64(prediction)
            
            # Return the results
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "message": "Successfully processed image",
                    "original": original_base64,
                    "ground_truth": ground_truth_base64,
                    "prediction": prediction_base64,
                    "image_path": image_path
                }),
                mimetype="application/json",
                status_code=200
            )
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": f"Error processing image: {str(e)}"
                }),
                mimetype="application/json",
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"Error in function: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"Error in function: {str(e)}"
            }),
            mimetype="application/json",
            status_code=500
        )
