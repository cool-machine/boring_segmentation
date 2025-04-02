import logging
import azure.functions as func
import json
import base64

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
    try:
        # Create a simple 1x1 pixel image in base64 format
        # This is a valid base64 representation of a tiny transparent PNG
        tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        
        # Return a minimal static response with valid base64 data
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "This is a minimal test response with valid base64 data",
                "original": tiny_png_base64,
                "ground_truth": tiny_png_base64,
                "prediction": tiny_png_base64,
                "image_path": "test_image.jpg"
            }),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error in minimal GetPrediction function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "traceback": tb}),
            mimetype="application/json",
            status_code=500
        )
