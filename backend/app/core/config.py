import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file in backend directory
load_dotenv(BASE_DIR / ".env")

PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "datasets"
MODEL_DIR = PROJECT_DIR / "trained_models"
PREPROCESSOR_DIR = MODEL_DIR / "preprocessors"
TRAINING_DATA_DIR = MODEL_DIR / "training_data"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API Key configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
