import os
from dotenv import load_dotenv

load_dotenv()

USE_OPENAI = True
USE_OLLAMA = False
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

TOP_K = 4
MODEL_NAME = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")