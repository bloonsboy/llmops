import os
from google.cloud import aiplatform
from google.cloud import storage
from google.api_core import exceptions
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_gcp_setup():
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION")
    bucket_name = os.getenv("GCP_BUCKET_NAME")

    if not project_id or not region or not bucket_name:
        logging.error("Please ensure GCP_PROJECT_ID, GCP_REGION, and GCP_BUCKET_NAME are set in your environment variables.")
        return

    logging.info("--- 1. Testing Vertex AI Connection ---")
    try:
        aiplatform.init(project=project_id, location=region)
        logging.info("Vertex AI client initialized successfully.")
        logging.info(f"Connected to project '{project_id}' in region '{region}'.\n")
    except exceptions.PermissionDenied:
        logging.error(f"Permission denied. The authenticated user/service account does not have the 'aiplatform.user' role on project '{project_id}'.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while initializing Vertex AI: {e}\n")
        return

    logging.info("--- 2. Testing Google Cloud Storage Connection ---")
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.get_bucket(bucket_name)
        logging.info(f"Successfully accessed bucket '{bucket_name}'.")

        blobs = list(storage_client.list_blobs(bucket, max_results=5))
        logging.info(f"Found {len(blobs)} files in the bucket (listing up to 5).")
        for blob in blobs:
            logging.info(f"{blob.name}")
        logging.info("GCS setup is correct.\n")

    except exceptions.NotFound:
        logging.error(f"Bucket '{bucket_name}' not found in project '{project_id}'. Please check the name.")
    except exceptions.Forbidden as e:
        logging.error(e)
        # logging.error(f"Permission denied. The authenticated user/service account does not have the 'Storage Object Viewer' role for the bucket '{bucket_name}'.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while accessing GCS: {e}\n")

if __name__ == "__main__":
    check_gcp_setup()
