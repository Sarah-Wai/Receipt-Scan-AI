# backend/config.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

#backend/db/receipts.db
DB_PATH = BASE_DIR / "db" / "receipts.db"


#backend/data
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

#backend/data/uploads/processed
PROCESSED_UPLOAD_DIR = UPLOAD_DIR / "processed"

RUN_MODE="DEVELOPMENT"  # or "PRODUCTION"


FRONTEND_ORIGIN = "http://localhost:4200"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

#GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY="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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

#GEMINI_API_KEY = "AIzaSyAMe-yvU-0taj9ZJK1p_l6QUTUJH4kPaqw" 
GEMINI_API_KEY = "AIzaSyCQGz4Pdl0n_qandmJKnkuPZg6dCsZXbfM"

GEMINI_TEXT_MODEL = "gemini-2.5-flash-lite"
GEMINI_VISION_MODEL = "gemini-2.5-flash-lite"

#Colab API URL
COLAB_LLM_WORKER_URL = "https://f1b5-34-125-208-88.ngrok-free.app"
COLAB_LLM_API_KEY = "demo-secret-key"