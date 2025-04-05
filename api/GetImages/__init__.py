# api/GetImages/__init__.py

import logging
import azure.functions as func
import json
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Log basic information
        logging.info("Minimal GetImages function executed")
        
        # Return a simple successful response
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "Minimal GetImages function executed successfully",
                "images": ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"],
                "container": "test",
                "source": "test"
            }),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        # Log any errors
        logging.error(f"Error in minimal GetImages function: {str(e)}")
        
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"Error in minimal GetImages function: {str(e)}"
            }),
            mimetype="application/json",
            status_code=500
        )