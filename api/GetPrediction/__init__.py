import logging
import azure.functions as func
import json
import base64
import os
import sys
import io
from PIL import Image
import numpy as np

# Add the api directory to the path so we can import from src
api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(api_dir)

# Import Azure utilities
from src.utils.azure_utils import download_blob_to_memory, get_blob_service_client

def image_to_base64(image):
    """Convert a PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def create_simple_segmentation(image):
    """
    Create a simple segmentation mask for demonstration purposes.
    In a real application, this would use a trained model.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: A simple segmentation mask
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create a simple mask based on color thresholds (this is just for demonstration)
    # In a real application, you would use your actual segmentation model here
    height, width = img_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Simple thresholding on the green channel for demonstration
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # Threshold on green channel
        green_channel = img_array[:, :, 1]
        mask[green_channel > 100] = 1  # Class 1 for green-ish areas
        mask[green_channel > 180] = 2  # Class 2 for very green areas
    
    # Convert back to PIL Image
    mask_image = Image.fromarray(mask.astype(np.uint8))
    return mask_image

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
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
            
            # Open the image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            logging.info(f"Image size: {image.size}, mode: {image.mode}")
            
            # Create a simple segmentation mask (in a real app, this would use your model)
            mask = create_simple_segmentation(image)
            logging.info(f"Created segmentation mask")
            
            # For demonstration, use the same mask for ground truth and prediction
            # In a real application, you would load the ground truth from storage
            # and generate the prediction using your model
            ground_truth = mask
            prediction = mask
            
            # Convert images to base64
            original_base64 = image_to_base64(image)
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
        except Exception as blob_error:
            logging.error(f"Error processing image: {str(blob_error)}")
            
            # Fallback to a simple 1x1 pixel image for testing
            tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            
            return func.HttpResponse(
                json.dumps({
                    "status": "partial_success",
                    "message": f"Error processing image: {str(blob_error)}. Using test image instead.",
                    "original": tiny_png_base64,
                    "ground_truth": tiny_png_base64,
                    "prediction": tiny_png_base64,
                    "image_path": image_path,
                    "error": str(blob_error)
                }),
                mimetype="application/json",
                status_code=200
            )
    except Exception as e:
        logging.error(f"Error in GetPrediction function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Traceback: {tb}")
        
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"Error in GetPrediction function: {str(e)}",
                "traceback": tb
            }),
            mimetype="application/json",
            status_code=500
        )
