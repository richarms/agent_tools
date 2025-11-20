import hashlib
import json
import os
import tempfile
import time
from typing import Optional, List, Dict, Any

from agent_tools import config
from agent_tools import utils

SESSION_STORE_DIR = os.path.join(tempfile.gettempdir(), "agent_tools_sessions")
SESSION_FILE_SUFFIX = ".json"


def _ensure_session_store_dir() -> Optional[str]:
    try:
        os.makedirs(SESSION_STORE_DIR, exist_ok=True)
    except OSError:
        return None
    return SESSION_STORE_DIR


def _session_file_path(session_id: str) -> Optional[str]:
    safe_id = (session_id or "").strip().lower()
    if not safe_id:
        return None
    if not all(char in "0123456789abcdef" for char in safe_id):
        return None
    directory = _ensure_session_store_dir()
    if not directory:
        return None
    return os.path.join(directory, f"{safe_id}{SESSION_FILE_SUFFIX}")


def _generate_session_id(payload: str) -> str:
    seed = f"{time.time_ns()}:{payload}".encode("utf-8", errors="ignore")
    return hashlib.sha256(seed).hexdigest()[:10]


def save_session_context(context: List[Dict[str, Any]]) -> Optional[str]:
    directory = _ensure_session_store_dir()
    if not directory:
        return None
    
    # Use local utils
    serialized_context = [utils._strip_status_fields(utils._json_safe_copy(item)) for item in context]
    
    scheme, host = config.get_target_info()
    
    payload = {
        "context": serialized_context,
        "saved_at": time.time(),
        "target_mc": {"scheme": scheme, "host": host},
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    session_id = _generate_session_id(payload_json)
    path = os.path.join(directory, f"{session_id}{SESSION_FILE_SUFFIX}")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload_json)
    except OSError:
        return None
    return session_id


def load_session(session_id: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Loads session and returns the context list. 
    Does NOT set the global context.
    """
    path = _session_file_path(session_id or "")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    
    saved_context = payload.get("context")
    if not isinstance(saved_context, list):
        return None
    
    saved_context = [utils._strip_status_fields(item) for item in saved_context]
    
    # Validate/fix system prompt if needed, though the caller usually handles initialization if empty
    return saved_context
