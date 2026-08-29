import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# BASE DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

QUESTIONS_DIR = DATA_DIR / "questions"
TEXTBOOKS_DIR = DATA_DIR / "textbooks"
ANSWER_SCRIPTS_DIR = DATA_DIR / "answer_scripts"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# ---------------------------------------------------------
# CREATE DIRECTORIES IF THEY DON'T EXIST
# ---------------------------------------------------------

for directory in [
    QUESTIONS_DIR,
    TEXTBOOKS_DIR,
    ANSWER_SCRIPTS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# AI CONFIGURATION
# ---------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

# ---------------------------------------------------------
# OCR CONFIGURATION
# ---------------------------------------------------------

OCR_LANGUAGE = "en"

# Minimum confidence used for OCR warnings
OCR_CONFIDENCE_THRESHOLD = 0.50

# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of textbook chunks to retrieve
TOP_K_CHUNKS = 5
ANSWER_KEYS_DIR = (
    BASE_DIR /
    "data" /
    "answer_keys"
)
