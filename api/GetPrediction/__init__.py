import logging
import azure.functions as func
import json
import base64
import io
from PIL import Image
import numpy as np

def encode_image_to_base64(image_array):
    """Convert a numpy array to a base64 encoded PNG image"""
    if image_array is None:
        return None
    
    # Create a PIL Image from the numpy array
    img = Image.fromarray(image_array.astype(np.uint8))
    
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Encode the bytes as base64
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64

def create_test_image(width=512, height=512, color=(100, 150, 200)):
    """Create a test image with the specified dimensions and color"""
    # Create a solid color image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    
    # Add some shapes for visual interest
    # Draw a rectangle
    img[100:200, 100:400, :] = (200, 100, 100)
    # Draw another rectangle
    img[300:400, 200:400, :] = (100, 200, 100)
    
    return img

def create_test_mask(width=512, height=512):
    """Create a test segmentation mask with the specified dimensions"""
    # Create a mask with different class values
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Class 1 (e.g., "human")
    mask[100:200, 100:400] = 1
    
    # Class 2 (e.g., "vehicle")
    mask[300:400, 200:400] = 2
    
    # Class 5 (e.g., "nature")
    mask[50:150, 50:150] = 5
    
    return mask

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
    try:
        # Parse request body
        logging.info("Parsing request body")
        req_body = req.get_json()
        
        # Get parameters
        image_path = req_body.get('image_path')
        
        if not image_path:
            return func.HttpResponse(
                json.dumps({"error": "No image path provided"}),
                mimetype="application/json",
                status_code=400
            )
            
        logging.info(f"Processing image: {image_path}")
        
        # Create test images and masks
        original_image = create_test_image(512, 512)
        ground_truth_mask = create_test_mask(512, 512)
        prediction_mask = create_test_mask(512, 512)  # In a real scenario, this would be different
        
        # Encode images to base64
        image_b64 = encode_image_to_base64(original_image)
        mask_b64 = encode_image_to_base64(np.stack([ground_truth_mask]*3, axis=-1) * 30)  # Multiply by 30 to make colors visible
        prediction_b64 = encode_image_to_base64(np.stack([prediction_mask]*3, axis=-1) * 30)  # Multiply by 30 to make colors visible
        
        # Return the mock response
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "This is a test response from the simplified GetPrediction function",
                "original": image_b64,
                "ground_truth": mask_b64,
                "prediction": prediction_b64,
                "image_path": image_path
            }),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error in simplified GetPrediction function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "traceback": tb}),
            mimetype="application/json",
            status_code=500
        )
