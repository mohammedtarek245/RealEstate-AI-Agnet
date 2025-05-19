"""
Configuration file for the Arabic Real Estate AI Agent.
Modified for Google Colab (flat file structure).
"""

import os
from enum import Enum
from pathlib import Path

# === Paths ===
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR
PROPERTIES_PATH = DATA_DIR / "properties.csv"
PHASE_KNOWLEDGE_DIR = DATA_DIR / "phase_knowledge"
RULES_PATH = DATA_DIR / "rules.json"
VECTOR_DB_PATH = DATA_DIR / "vector_db"

# === Agent behavior ===
DEFAULT_LANGUAGE = "ar"
DEFAULT_DIALECT = "Egyptian"
MAX_HISTORY_LENGTH = 10
TEMPERATURE = 0.7
TOP_P = 0.9

# === RAG configurations ===
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
TOP_K_RETRIEVAL = 3

# === UI configurations ===
GRADIO_THEME = "default"  # Avoid "dark" issue in Colab
UI_TITLE = "وكيل العقارات الذكي"
UI_DESCRIPTION = "مرحبًا بك في وكيل العقارات الذكي. يمكنني مساعدتك في العثور على العقار المناسب."
UI_WELCOME_MESSAGE = "اهلا بيك! انا وكيلك العقاري. تحب أبدأ ازاي؟"

# === Debugging ===
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# === Conversation Phase Enumeration ===
class ConversationPhase(Enum):
    DISCOVERY = 1
    SUMMARY = 2
    SUGGESTION = 3
    PERSUASION = 4
    ALTERNATIVE = 5
    URGENCY = 6
    CLOSING = 7
