import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR=os.getenv("DATA_DIR", "/data")
CREDITCARD_CSV=os.path.join(DATA_DIR, "creditcard.csv")

LOG_DIR=os.getenv("LOG_DIR", "/logs")
EXPERIMENT_LOG=os.path.join(LOG_DIR, "experiment_log.csv")

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "../logs")
GOOGLE_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")     # for Vertex AI / Gemini
GOOGLE_REGION = os.getenv("GOOGLE_REGION", "us-central1")

VECTOR_DB=os.getenv("VECTOR_DB", "chroma")
RANDOM_STATE=int(os.getenv("RANDOM_STATE", 42))

HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

LOCAL_MISTRAL_ID=os.getenv("LOCAL_MISTRAL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
#LOCAL_MISTRAL_ID=os.getenv("LOCAL_MISTRAL_ID", "mistralai/Mistral-7B-v0.1")
#LOCAL_MISTRAL_ID=os.getenv("LOCAL_MISTRAL_ID", "TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
LOCAL_LLAMA_ID=os.getenv("LOCAL_LLAMA_ID", "meta-llama/Llama-2-7b-chat-hf") 