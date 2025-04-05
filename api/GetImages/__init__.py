# api/GetImages/__init__.py

import logging
import azure.functions as func
import json
import os
import sys

# Add the api directory to the path so we can import from src
api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(api_dir)

# Import Azure utilities
from src.utils.azure_utils import list_blobs, get_blob_service_client

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Get container name from request or use default
        container_name = req.params.get('container_name', os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1"))
        logging.info(f"Using container name: {container_name}")
        
        # Get images path from request or use default
        images_path = req.params.get('images_path', os.environ.get("AZURE_IMAGES_PATH", "images"))
        logging.info(f"Using images path prefix: {images_path}")
        
        # List blobs in the container with the specified prefix
        try:
            # Log environment variables for debugging (redacting sensitive info)
            conn_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if conn_string:
                # Redact the account key for security
                redacted_conn = conn_string
                if "AccountKey=" in redacted_conn:
                    parts = redacted_conn.split("AccountKey=")
                    if len(parts) > 1 and ";" in parts[1]:
                        key_part = parts[1].split(";")[0]
                        redacted_conn = redacted_conn.replace(key_part, "REDACTED")
                logging.info(f"Using connection string: {redacted_conn}")
            else:
                logging.error("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
            
            logging.info(f"Environment variables:")
            for key in os.environ:
                if key.startswith("AZURE_") and "KEY" not in key.upper() and "SECRET" not in key.upper() and "PASSWORD" not in key.upper():
                    logging.info(f"  {key}: {os.environ[key]}")
            
            # Try to get the blob service client
            try:
                blob_service_client = get_blob_service_client()
                logging.info(f"Successfully created blob service client")
            except Exception as client_error:
                logging.error(f"Error creating blob service client: {str(client_error)}")
                raise
            
            # Try to list blobs
            blobs = list_blobs(prefix=images_path, container_name=container_name)
            
            # Extract image names from blobs
            image_names = [blob.name for blob in blobs]
            
            # Filter out non-image files (keep only .jpg, .jpeg, .png)
            image_extensions = ['.jpg', '.jpeg', '.png']
            filtered_images = [img for img in image_names if any(img.lower().endswith(ext) for ext in image_extensions)]
            
            logging.info(f"Found {len(filtered_images)} images in container {container_name}")
            
            # Return the list of images
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "message": f"Retrieved {len(filtered_images)} images from Azure Blob Storage",
                    "images": filtered_images,
                    "container": container_name,
                    "source": "azure_storage"
                }),
                mimetype="application/json",
                status_code=200
            )
        except Exception as blob_error:
            logging.error(f"Error listing blobs: {str(blob_error)}")
            logging.warning("Falling back to test images")
            
            # Fallback to test images if blob listing fails
            return func.HttpResponse(
                json.dumps({
                    "status": "partial_success",
                    "message": "Failed to retrieve images from Azure Blob Storage. Using test images instead.",
                    "images": ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"],
                    "container": container_name,
                    "source": "test",
                    "error": str(blob_error)
                }),
                mimetype="application/json",
                status_code=200
            )

    except Exception as e:
        logging.error(f"Error in GetImages function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Traceback: {tb}")
        
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"Error in GetImages function: {str(e)}",
                "traceback": tb
            }),
            mimetype="application/json",
            status_code=500
        )