# test_azure_connection.py
import os
import sys
import logging
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the api directory to the path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.join(current_dir, "api")
sys.path.append(api_dir)

# Try to import Azure utilities
try:
    from src.utils.azure_utils import get_blob_service_client, get_container_client, list_blobs
    logger.info("Successfully imported Azure utilities")
except ImportError as e:
    logger.error(f"Failed to import Azure utilities: {str(e)}")
    sys.exit(1)

def test_connection_string():
    """Test if the Azure Storage connection string is valid"""
    logger.info("Testing Azure Storage connection string...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if connection string exists
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        logger.error("AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        return False
    
    # Try to connect using the connection string
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        logger.info("Successfully connected to Azure Storage using connection string")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Azure Storage: {str(e)}")
        return False

def test_container_exists(container_name):
    """Test if the specified container exists"""
    logger.info(f"Testing if container '{container_name}' exists...")
    
    try:
        # Get blob service client
        blob_service_client = get_blob_service_client()
        
        # List containers
        containers = [container.name for container in blob_service_client.list_containers()]
        
        if container_name in containers:
            logger.info(f"Container '{container_name}' exists")
            return True
        else:
            logger.error(f"Container '{container_name}' does not exist")
            logger.info(f"Available containers: {containers}")
            return False
    except Exception as e:
        logger.error(f"Error checking container: {str(e)}")
        return False

def test_images_path(container_name, images_path):
    """Test if images exist at the specified path in the container"""
    logger.info(f"Testing if images exist at path '{images_path}' in container '{container_name}'...")
    
    try:
        # List blobs with the specified prefix
        blobs = list_blobs(prefix=images_path, container_name=container_name)
        
        if blobs:
            logger.info(f"Found {len(blobs)} blobs at path '{images_path}'")
            logger.info(f"First few blobs: {[blob.name for blob in blobs[:5]]}")
            return True
        else:
            logger.error(f"No blobs found at path '{images_path}'")
            return False
    except Exception as e:
        logger.error(f"Error checking images path: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Azure Storage connection tests...")
    
    # Load environment variables
    load_dotenv()
    
    # Get container and path information from environment variables
    container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
    images_path = os.environ.get("AZURE_IMAGES_PATH", "images")
    
    # Test connection string
    if not test_connection_string():
        logger.error("Connection string test failed. Please check your AZURE_STORAGE_CONNECTION_STRING")
        return
    
    # Test container exists
    if not test_container_exists(container_name):
        logger.error(f"Container test failed. Please check if '{container_name}' exists")
        return
    
    # Test images path
    if not test_images_path(container_name, images_path):
        logger.error(f"Images path test failed. Please check if '{images_path}' exists in container '{container_name}'")
        return
    
    logger.info("All tests passed successfully!")
    logger.info("Your Azure Storage connection and paths are valid")

if __name__ == "__main__":
    main()