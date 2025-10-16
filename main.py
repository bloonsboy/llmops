import os
from google.cloud import aiplatform, storage


def test_gcp_setup():
    """Teste rapidement la configuration de Vertex AI et GCS."""
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION")
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not all([project_id, region, bucket_name]):
        print(
            "Erreur : Définir les variables GCP_PROJECT_ID, GCP_REGION, et GCS_BUCKET_NAME."
        )
        return

    print(f"--- Test de la configuration pour le projet '{project_id}' ---")

    try:
        aiplatform.init(project=project_id, location=region)
        print("[SUCCÈS] Client Vertex AI initialisé.")
    except Exception as e:
        print(f"[ÉCHEC] Initialisation de Vertex AI : {e}")
        return

    try:
        storage.Client(project=project_id).get_bucket(bucket_name)
        print(f"[SUCCÈS] Le bucket GCS '{bucket_name}' est accessible.")
    except Exception as e:
        print(f"[ÉCHEC] Accès à GCS : {e}")

    print("--- Test terminé ---")


if __name__ == "__main__":
    test_gcp_setup()
