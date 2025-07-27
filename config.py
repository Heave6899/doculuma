# config.py
import logging
from pathlib import Path
from datetime import datetime, timezone
import re
# ─── Constants ──────────────────────────────────────────────────────────────────
DB_PATH = Path("data.db")
LLM_MODEL_PATH = "./models/sqlcoder-7b-q5_k_m.gguf"
MAX_PREVIEW_ROWS = 1000
TABLE_NAME_REGEX = r'^[A-Za-z_]\w*$'
SUPPORTED_EXTS = {"xlsx", "xls", "csv", "ods", "json"}
PRUNE_KEEP = 5  # Keep last N snapshots per base table

UNQORK_MODULE_FOLDER = Path("data_saves/unqork_modules") # Dedicated folder for Unqork module JSONs
UNQORK_WORKSPACE_EXTRACT_OUTPUT = Path("data_saves/unqork_workspace_extracts") # For extracted "active" Unqork modules
UNQORK_EXTRACTOR_OUTPUT_FILE_NAME = 'unqork_module_definitions_with_deps.xlsx'

UNQORK_REMOTE_ALL_MODULES_URL = "https://nyl-uapp-staging.unqork.io/fbu/uapi/modules"
UNQORK_REMOTE_SINGLE_MODULE_URL = "https://nyl-uapp-staging.unqork.io/fbu/form/" # Kept for consistency with original, though /uapi/modules/{id} is preferred

# Default Bearer token (can be overridden by Streamlit input)
UNQORK_API_BEARER_TOKEN = None

# Which field 'type' values to treat as UI inputs for Unqork modules
UNQORK_UI_INPUT_TYPES = [
    "radio", "text", "textarea", "number", "email", "checkbox",
    "textfield", "basicDropdown", "dropdown", "dateinput",
    "currency", "toggle", "button", "richtext", "html",
    "files", "image", "signature", "address", "phone", "datagrid", "content"
]

# For Unqork Workspace Extractor (to define "real active" modules)
UNQORK_MODIFIED_AFTER = datetime(2024, 12, 5, tzinfo=timezone.utc) # Current date is ~2025, so this filters older modules
UNQORK_EXCLUDE_PATTERNS = re.compile(r'\b(poc|test|copy|dev|zzz|backup|old)\b', re.IGNORECASE)

CHROMA_COLLECTION_PREFIX = "doculuma_embeddings"

# ─── Logging Configuration ──────────────────────────────────────────────────────
# This setup runs once when the module is imported.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"), # Log to a file
        logging.StreamHandler()         # Log to the console
    ]
)
logger = logging.getLogger(__name__)