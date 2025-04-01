# api/GetImages/__init__.py

import logging
import azure.functions as func
import json
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Get container name from request or use default
        container_name = req.params.get('container_name', os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1"))
        
        # Return a test response with hardcoded images
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "This is a test response from the simplified GetImages function",
                "images": ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"],
                "container": container_name,
                "source": "test"
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error in simplified GetImages function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "traceback": tb}),
            mimetype="application/json",
            status_code=500
        )