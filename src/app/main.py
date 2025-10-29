import re
import subprocess
import re
import subprocess  # Gardez-le pour l'instant, même si on ne l'utilise plus
import google.auth
import google.auth.transport.requests
import chainlit as cl
import requests
# ... (le reste de vos imports)
from chainlit.message import Message
from transformers import AutoTokenizer

from src.constants import ENDPOINT_ID, PROJECT_NUMBER

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
ENDPOINT_URL = f"https://europe-west1-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/europe-west1/endpoints/{ENDPOINT_ID}:predict"

@cl.set_starters  # type: ignore
async def set_starters():
    """Set starter messages for the Chainlit app."""
    return [
        cl.Starter(
            label="Message #1 - Lightsaber talk",
            message="Paint the handle of your lightsaber on your hip dull gray.",
        ),
        cl.Starter(
            label="Message #2 - Not very nice",
            message="He is not very nice.",
        ),
        cl.Starter(
            label="Message #3 - Motivational quote",
            message="You must believe in the force.",
        ),
    ]


@cl.on_message
async def handle_message(message: Message):
    """Handle incoming messages from the user."""
    await cl.Message(content=call_model_api(message)).send()


def build_prompt(tokenizer: AutoTokenizer, sentence: str):
    """Build a prompt from a sentence applying the chat template."""
    return tokenizer.apply_chat_template(  # type: ignore
        [
            {"role": "user", "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_response(generated_text: str) -> str:
    """Extract the model's response from the generated text."""
    return re.findall(
        r"(?:<\|assistant\|>)([^<]*)",
        generated_text,
    )[0]

def call_model_api(message: Message) -> str:
    """Call the custom LLM chat model API."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

    # --- NOUVELLE MÉTHODE D'AUTH ---
    try:
        credentials, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
    except Exception as e:
        print(f"ERREUR D'AUTHENTIFICATION: {e}")
        return "Désolé, je n'ai pas pu m'authentifier auprès de Google."
    # -------------------------------

    templated_input = build_prompt(tokenizer, message.content)
    model_input = {
        "instances": [{"input": templated_input}],
        "parameters": {
            "maxOutputTokens": 64,
            "temperature": 0.1,
            "topP": 0.8,
        },
    }

    # --- APPEL API ROBUSTE ---
    raw_response = requests.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=model_input,
    )

    if raw_response.status_code != 200:
        print(f"ERREUR DE L'API (Statut {raw_response.status_code}):")
        print(raw_response.text)
        return "Désolé, une erreur est survenue lors de l'appel du modèle."

    response_json = raw_response.json()

    if "predictions" not in response_json:
        print("ERREUR: La réponse JSON ne contient pas la clé 'predictions'")
        print(response_json)
        return "Désolé, la réponse du modèle est invalide."

    raw_model_response = response_json["predictions"][0]
    
    try:
        extracted_response = extract_response(raw_model_response)
        return extracted_response
    except IndexError:
        print(f"ERREUR: La regex n'a rien trouvé dans la réponse: {raw_model_response}")
        return "Désolé, je n'ai pas pu extraire de réponse."
    
