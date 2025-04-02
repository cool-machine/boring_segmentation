import logging
import azure.functions as func
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
    try:
        # Return a minimal static response
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "This is a minimal test response",
                "original": "test_image_data",
                "ground_truth": "test_mask_data",
                "prediction": "test_prediction_data",
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
