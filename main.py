import os
from google.cloud import aiplatform, storage

def test_gcp_setup():
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION")
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not all([project_id, region, bucket_name]):
        print(
            "Error: Please set the GCP_PROJECT_ID, GCP_REGION, and GCS_BUCKET_NAME environment variables."
        )
        return

    print(f"--- Testing configuration for project '{project_id}' ---")

    try:
        aiplatform.init(project=project_id, location=region)
        print("[SUCCESS] Vertex AI client initialized.")
    except Exception as e:
        print(f"[FAILURE] Vertex AI initialization: {e}")
        return

    try:
        storage.Client(project=project_id).get_bucket(bucket_name)
        print(f"[SUCCESS] GCS bucket '{bucket_name}' is accessible.")
    except Exception as e:
        print(f"[FAILURE] GCS access: {e}")

    print("--- Test completed ---")


if __name__ == "__main__":
    test_gcp_setup()
