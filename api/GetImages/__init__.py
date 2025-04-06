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
        
        # Get pip freeze output
        pip_freeze = []
        try:
            result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
            pip_freeze = result.stdout.strip().split('\n') if result.stdout else []
        except Exception as e:
            pip_freeze = [f"Error running pip freeze: {str(e)}"]
        
        # Get diagnostics
        diagnostics = {}
        diagnostics["python_version"] = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        diagnostics["os"] = os.sys.platform
        
        # Get environment diagnostics
        env_diagnostics = {}
        env_diagnostics["environment_variables"] = dict(os.environ)
        
        # Check if this is a package verification request
        verify_packages = req.params.get('verify_packages')
        if verify_packages and verify_packages.lower() == 'true':
            # Return only package status for verification
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "message": "Package verification completed",
                    "package_status": package_status,
                    "diagnostics": diagnostics,
                    "environment": env_diagnostics,
                    "pip_freeze": pip_freeze
                }),
                mimetype="application/json",
                status_code=200
            )
        
        # Return a simple successful response with test images and package status
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "Simplified GetImages function executed successfully",
                "images": ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"],
                "container": "test",
                "source": "test",
                "package_status": package_status,
                "diagnostics": diagnostics,
                "environment": env_diagnostics,
                "pip_freeze": pip_freeze
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