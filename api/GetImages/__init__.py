# api/GetImages/__init__.py

import logging
import azure.functions as func
import json
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Log basic information
        logging.info("Simplified GetImages function executed")
        
        # Check if packages are available by trying to import them
        package_status = {}
        try:
            import azure.storage.blob
            package_status["azure.storage.blob"] = "Available"
        except ImportError as e:
            package_status["azure.storage.blob"] = f"Not available: {str(e)}"
        
        try:
            import numpy
            package_status["numpy"] = "Available"
        except ImportError as e:
            package_status["numpy"] = f"Not available: {str(e)}"
        
        # Return a simple successful response with test images and package status
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "Simplified GetImages function executed successfully",
                "images": ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"],
                "container": "test",
                "source": "test",
                "package_status": package_status
            }),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        # Log any errors
        logging.error(f"Error in simplified GetImages function: {str(e)}")
        
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"Error in simplified GetImages function: {str(e)}"
            }),
            mimetype="application/json",
            status_code=500
        )