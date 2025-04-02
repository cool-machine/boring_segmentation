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
        try:
            req_body = req.get_json()
            logging.info(f"Request body parsed successfully: {req_body}")
        except ValueError:
            logging.error("Invalid JSON in request body")
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Get parameters
        image_path = req_body.get('image_path')
        logging.info(f"Image path from request: {image_path}")
        
        if not image_path:
            logging.warning("No image path provided in request")
            return func.HttpResponse(
                json.dumps({"error": "No image path provided"}),
                mimetype="application/json",
                status_code=400
            )
            
        logging.info(f"Processing image: {image_path}")
        
        try:
            # Create test images and masks
            logging.info("Creating test image")
            original_image = create_test_image(512, 512)
            logging.info("Creating test ground truth mask")
            ground_truth_mask = create_test_mask(512, 512)
            logging.info("Creating test prediction mask")
            prediction_mask = create_test_mask(512, 512)  # In a real scenario, this would be different
            
            # Encode images to base64
            logging.info("Encoding original image to base64")
            image_b64 = encode_image_to_base64(original_image)
            logging.info("Encoding ground truth mask to base64")
            mask_b64 = encode_image_to_base64(np.stack([ground_truth_mask]*3, axis=-1) * 30)  # Multiply by 30 to make colors visible
            logging.info("Encoding prediction mask to base64")
            prediction_b64 = encode_image_to_base64(np.stack([prediction_mask]*3, axis=-1) * 30)  # Multiply by 30 to make colors visible
        except Exception as img_error:
            logging.error(f"Error creating or encoding images: {str(img_error)}")
            import traceback
            img_tb = traceback.format_exc()
            logging.error(f"Image processing traceback: {img_tb}")
            return func.HttpResponse(
                json.dumps({"error": f"Error processing images: {str(img_error)}"}),
                mimetype="application/json",
                status_code=500
            )
        
        # Prepare response
        logging.info("Preparing response JSON")
        response_data = {
            "status": "success",
            "message": "This is a test response from the simplified GetPrediction function",
            "original": image_b64,
            "ground_truth": mask_b64,
            "prediction": prediction_b64,
            "image_path": image_path
        }
        
        # Return the mock response
        logging.info("Returning successful response")
        return func.HttpResponse(
            json.dumps(response_data),
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
