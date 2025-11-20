import os
import tempfile
from typing import Optional, Tuple
from urllib.parse import urlparse

SYSTEM_PROMPT = ("You are a powerful assistant. Use all the tools to which you have access to answer queries about the MeerKAT telescope infrastructure system health and status. Your primary aim is to highlight anomolies and find errors.")

DEFAULT_MODEL = "gpt-5-mini"

DEFAULT_TARGET_MC = "http://lab-mc.sdp.kat.ac.za"
_DEFAULT_TARGET = urlparse(DEFAULT_TARGET_MC)
_DEFAULT_SCHEME = _DEFAULT_TARGET.scheme or "http"
_DEFAULT_HOST = _DEFAULT_TARGET.hostname or _DEFAULT_TARGET.netloc or "lab-mc.sdp.kat.ac.za"

# Globals
_target_scheme, _target_host = _DEFAULT_SCHEME, _DEFAULT_HOST
_model_name = DEFAULT_MODEL


def _normalize_target_mc(value: Optional[str]) -> Tuple[str, str]:
    candidate = (value or "").strip()
    if not candidate:
        candidate = DEFAULT_TARGET_MC
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    scheme = parsed.scheme or _DEFAULT_SCHEME
    host = parsed.hostname or _DEFAULT_HOST
    return scheme, host


def configure_target_mc(value: Optional[str]) -> None:
    global _target_scheme, _target_host
    _target_scheme, _target_host = _normalize_target_mc(value)


def configure_model(value: Optional[str]) -> None:
    global _model_name
    candidate = (value or "").strip()
    _model_name = candidate or DEFAULT_MODEL


def get_model_name() -> str:
    return _model_name


def get_target_info() -> Tuple[str, str]:
    return _target_scheme, _target_host


def mc_service_url(port: int, suffix: str = "") -> str:
    host = _target_host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    normalized_suffix = suffix or ""
    if normalized_suffix and not normalized_suffix.startswith("/"):
        normalized_suffix = f"/{normalized_suffix}"
    return f"{_target_scheme}://{host}:{port}{normalized_suffix}"
