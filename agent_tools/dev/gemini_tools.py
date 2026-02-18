import argparse
import base64
import json
import os
import subprocess
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import google.generativeai as genai
from google.generativeai import types as genai_types
from google.protobuf.json_format import MessageToDict
import requests
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a powerful assistant. When the user asks for up-to-date information beyond your training date, or "
    "explicitly requests a Google search, you MUST call the `google_search` tool before responding. Use the search "
    "results to craft your answer. If the tool fails, explain the failure and provide next steps. If the user is only "
    "interested in whether a website is up, preferentially use the ping tool."
)

ToolHandler = Callable[..., str]


DEFAULT_TARGET_MC = "http://lab-mc.sdp.kat.ac.za"
_DEFAULT_TARGET = urlparse(DEFAULT_TARGET_MC)
_DEFAULT_SCHEME = _DEFAULT_TARGET.scheme or "http"
_DEFAULT_HOST = _DEFAULT_TARGET.hostname or _DEFAULT_TARGET.netloc or "lab-mc.sdp.kat.ac.za"


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


_target_scheme, _target_host = _normalize_target_mc(DEFAULT_TARGET_MC)


def configure_target_mc(value: Optional[str]) -> None:
    global _target_scheme, _target_host
    _target_scheme, _target_host = _normalize_target_mc(value)


def mc_service_url(port: int, suffix: str = "") -> str:
    host = _target_host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    normalized_suffix = suffix or ""
    if normalized_suffix and not normalized_suffix.startswith("/"):
        normalized_suffix = f"/{normalized_suffix}"
    return f"{_target_scheme}://{host}:{port}{normalized_suffix}"


configure_target_mc(os.getenv("TARGET_MC"))


GEMINI_SCHEMA_BLOCKED_KEYS = {"additionalProperties"}


def sanitize_schema_for_gemini(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, val in value.items():
            if key in GEMINI_SCHEMA_BLOCKED_KEYS:
                continue
            sanitized[key] = sanitize_schema_for_gemini(val)
        return sanitized
    if isinstance(value, list):
        return [sanitize_schema_for_gemini(item) for item in value]
    return value


def message_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        try:
            return dict(value.items())
        except Exception:
            pass
    try:
        return MessageToDict(value)
    except Exception:
        return {}


@dataclass
class UsageTotals:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "UsageTotals") -> None:
        self.total_tokens += other.total_tokens
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens

    def reset(self) -> None:
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.reasoning_tokens = 0

    def has_data(self) -> bool:
        return any(
            (
                self.total_tokens,
                self.input_tokens,
                self.output_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
            )
        )

    @staticmethod
    def _as_int(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        return int(value) if isinstance(value, (int, float)) else 0

    @staticmethod
    def _get_attr(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @classmethod
    def from_usage(cls, usage: Any) -> Optional["UsageTotals"]:
        if usage is None:
            return None
        input_tokens = cls._as_int(cls._get_attr(usage, "input_tokens"))
        output_tokens = cls._as_int(cls._get_attr(usage, "output_tokens"))
        total_tokens = cls._as_int(cls._get_attr(usage, "total_tokens"))
        input_details = cls._get_attr(usage, "input_tokens_details")
        cached_input = cls._as_int(cls._get_attr(input_details, "cached_tokens"))
        output_details = cls._get_attr(usage, "output_tokens_details")
        reasoning_tokens = cls._as_int(cls._get_attr(output_details, "reasoning_tokens"))
        if not total_tokens:
            total_tokens = input_tokens + output_tokens
        if not (total_tokens or input_tokens or output_tokens or cached_input or reasoning_tokens):
            return None
        return cls(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input,
            reasoning_tokens=reasoning_tokens,
        )


class TokenTracker:
    def __init__(self) -> None:
        self._turn_totals = UsageTotals()
        self._session_totals = UsageTotals()

    def start_turn(self) -> None:
        self._turn_totals.reset()

    def record(self, usage: Any) -> None:
        totals = UsageTotals.from_usage(usage)
        if totals is None:
            return
        self._turn_totals.add(totals)
        self._session_totals.add(totals)

    def end_turn(self) -> Optional[str]:
        if not self._turn_totals.has_data():
            return None
        lines = [format_usage("Token usage", self._turn_totals)]
        if self._session_totals.total_tokens > self._turn_totals.total_tokens:
            lines.append(format_usage("Session total", self._session_totals))
        self._turn_totals.reset()
        return "\n".join(lines)

    def session_summary(self) -> Optional[str]:
        if not self._session_totals.has_data():
            return None
        return format_usage("Session total", self._session_totals)


def format_number(value: int) -> str:
    return f"{value:,}"


def format_usage(label: str, totals: UsageTotals) -> str:
    parts = [f"{label}: total={format_number(totals.total_tokens)}"]
    if totals.input_tokens or totals.cached_input_tokens:
        segment = f"input={format_number(totals.input_tokens)}"
        if totals.cached_input_tokens:
            segment += f" (+ {format_number(totals.cached_input_tokens)} cached)"
        parts.append(segment)
    if totals.output_tokens:
        segment = f"output={format_number(totals.output_tokens)}"
        if totals.reasoning_tokens:
            segment += f" (reasoning {format_number(totals.reasoning_tokens)})"
        parts.append(segment)
    return " ".join(parts)


token_tracker = TokenTracker()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler

    def as_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def as_gemini_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": sanitize_schema_for_gemini(self.parameters),
        }


def safe_tool(func: ToolHandler) -> ToolHandler:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return f"error: {exc}"

    return wrapper


@safe_tool
def ping(host: str = "") -> str:
    result = subprocess.run(
        ["ping", "-c", "5", host],
        text=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        check=False,
    )
    return result.stdout


@safe_tool
def google_search(query: str = "", num_results: int = 3) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cx:
        return "error: GOOGLE_API_KEY and GOOGLE_CSE_ID must be set"
    num = max(1, min(int(num_results or 3), 10))
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": num},
        timeout=10,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        return "no results"
    lines = []
    for item in items:
        title = item.get("title", "untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"{title}\n{link}\n{snippet}")
    return "\n\n".join(lines)


@safe_tool
def http_request(
    url: str = "",
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    if not url:
        return "error: url is required"
    allowed_methods = {"GET", "HEAD", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"
    try:
        timeout_seconds = max(1, min(int(timeout or 10), 30))
    except ValueError:
        return "error: timeout must be an integer"
    try:
        response = requests.request(
            verb,
            url,
            headers=headers,
            data=body,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        return f"error: {exc}"

    lines = [
        f"URL: {response.url}",
        f"Status: {response.status_code} {response.reason}",
        f"Elapsed: {response.elapsed.total_seconds():.2f}s",
    ]

    header_items = list(response.headers.items())
    if header_items:
        lines.append("Headers:")
        for key, value in header_items[:15]:
            lines.append(f"{key}: {value}")
        if len(header_items) > 15:
            lines.append(f"... {len(header_items) - 15} more headers omitted ...")

    if verb != "HEAD":
        text = response.text or ""
        max_chars = 2000
        snippet = text[:max_chars]
        lines.append("Body:")
        lines.append(snippet)
        if len(text) > max_chars:
            lines.append(f"... truncated {len(text) - max_chars} characters ...")

    return "\n".join(lines)


@safe_tool
def grafana_request(
    path: str = "",
    method: str = "GET",
    query: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    if not path:
        return "error: path is required"

    token = os.getenv("GRAFANA_API_TOKEN")
    if not token:
        return "error: GRAFANA_API_TOKEN must be set"

    base_url = os.getenv("GRAFANA_API_URL", mc_service_url(3000, "/api"))
    allowed_methods = {"GET", "HEAD", "POST", "PUT", "DELETE"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"

    try:
        timeout_seconds = max(1, min(int(timeout or 10), 30))
    except ValueError:
        return "error: timeout must be an integer"

    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    request_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if body:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    try:
        response = requests.request(
            verb,
            url,
            params=query,
            data=body,
            headers=request_headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return f"error: {exc}"

    lines = [
        f"URL: {response.url}",
        f"Status: {response.status_code} {response.reason}",
        f"Elapsed: {response.elapsed.total_seconds():.2f}s",
    ]

    try:
        payload = response.json()
        pretty = json.dumps(payload, indent=2)
        lines.append("JSON:")
        lines.append(pretty[:4000])
        if len(pretty) > 4000:
            lines.append(f"... truncated {len(pretty) - 4000} characters ...")
    except (ValueError, TypeError):
        text = response.text or ""
        lines.append("Body:")
        snippet = text[:2000]
        lines.append(snippet)
        if len(text) > 2000:
            lines.append(f"... truncated {len(text) - 2000} characters ...")

    return "\n".join(lines)


READ_ONLY_POST_PREFIXES = (
    "/_search",
    "/_msearch",
    "/_count",
    "/_field_caps",
    "/_sql",
    "/_explain",
)

WRITE_PATH_KEYWORDS = (
    "_delete",
    "_create",
    "_update",
    "_bulk",
    "_reindex",
    "_rollover",
    "_snapshot",
    "_shrink",
    "_split",
    "_forcemerge",
    "_close",
    "_open",
    "_ilm",
    "_ingest",
    "_tasks",
    "_scripts",
    "_template",
    "_component_template",
    "_index_template",
    "/_doc",
    "_security",
    "_license",
    "_watcher",
)


def _is_read_only_path(path: str) -> bool:
    normalized = "/" + path.lstrip("/")
    if any(keyword in normalized for keyword in WRITE_PATH_KEYWORDS):
        return False
    return True


def _is_allowed_post_path(path: str) -> bool:
    normalized = "/" + path.lstrip("/")
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in READ_ONLY_POST_PREFIXES
    )


@safe_tool
def kibana_request(
    path: str = "",
    method: str = "GET",
    query: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    if not path:
        return "error: path is required"

    api_key = os.getenv("KIBANA_API_KEY")
    basic_auth = os.getenv("KIBANA_BASIC_AUTH")
    allow_anonymous = os.getenv("KIBANA_ALLOW_NO_AUTH")
    allow_anonymous = (
        str(allow_anonymous).strip().lower() in {"1", "true", "yes", "on"}
        if allow_anonymous is not None
        else False
    )
    if api_key:
        auth_header = f"ApiKey {api_key}"
    elif basic_auth:
        encoded = base64.b64encode(basic_auth.encode("utf-8")).decode("utf-8")
        auth_header = f"Basic {encoded}"
    elif allow_anonymous:
        auth_header = None
    else:
        return "error: set KIBANA_API_KEY or KIBANA_BASIC_AUTH ('username:password')"

    base_url = os.getenv("KIBANA_API_URL", mc_service_url(5601))
    allowed_methods = {"GET", "HEAD", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"
    if not _is_read_only_path(path):
        return "error: path blocked to prevent write operations"
    if verb == "POST" and not _is_allowed_post_path(path):
        return "error: POST is limited to read-only endpoints such as *_search"

    try:
        timeout_seconds = max(1, min(int(timeout or 10), 30))
    except ValueError:
        return "error: timeout must be an integer"

    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    request_headers = {
        "Accept": "application/json",
        "kbn-xsrf": "true",
    }
    if auth_header:
        request_headers["Authorization"] = auth_header
    if body:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    try:
        response = requests.request(
            verb,
            url,
            params=query,
            data=body,
            headers=request_headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return f"error: {exc}"

    lines = [
        f"URL: {response.url}",
        f"Status: {response.status_code} {response.reason}",
        f"Elapsed: {response.elapsed.total_seconds():.2f}s",
    ]

    try:
        payload = response.json()
        pretty = json.dumps(payload, indent=2)
        lines.append("JSON:")
        lines.append(pretty[:4000])
        if len(pretty) > 4000:
            lines.append(f"... truncated {len(pretty) - 4000} characters ...")
    except (ValueError, TypeError):
        text = response.text or ""
        lines.append("Body:")
        snippet = text[:2000]
        lines.append(snippet)
        if len(text) > 2000:
            lines.append(f"... truncated {len(text) - 2000} characters ...")

    return "\n".join(lines)


@safe_tool
def logtrail_request(
    path: str = "default/json",
    method: str = "GET",
    query: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    api_key = os.getenv("KIBANA_API_KEY")
    basic_auth = os.getenv("KIBANA_BASIC_AUTH")
    allow_anonymous = os.getenv("KIBANA_ALLOW_NO_AUTH")
    allow_anonymous = (
        str(allow_anonymous).strip().lower() in {"1", "true", "yes", "on"}
        if allow_anonymous is not None
        else False
    )
    if api_key:
        auth_header = f"ApiKey {api_key}"
    elif basic_auth:
        encoded = base64.b64encode(basic_auth.encode("utf-8")).decode("utf-8")
        auth_header = f"Basic {encoded}"
    elif allow_anonymous:
        auth_header = None
    else:
        return "error: set KIBANA_API_KEY or KIBANA_BASIC_AUTH ('username:password')"

    base_url = os.getenv("LOGTRAIL_API_URL", mc_service_url(5601, "/logtrail"))
    allowed_methods = {"GET", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"

    try:
        timeout_seconds = max(1, min(int(timeout or 10), 30))
    except ValueError:
        return "error: timeout must be an integer"

    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    request_headers = {
        "Accept": "application/json",
        "kbn-xsrf": "true",
    }
    if auth_header:
        request_headers["Authorization"] = auth_header
    if body:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    try:
        response = requests.request(
            verb,
            url,
            params=query,
            data=body,
            headers=request_headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return f"error: {exc}"

    lines = [
        f"URL: {response.url}",
        f"Status: {response.status_code} {response.reason}",
        f"Elapsed: {response.elapsed.total_seconds():.2f}s",
    ]

    try:
        payload = response.json()
        pretty = json.dumps(payload, indent=2)
        lines.append("JSON:")
        lines.append(pretty[:4000])
        if len(pretty) > 4000:
            lines.append(f"... truncated {len(pretty) - 4000} characters ...")
    except (ValueError, TypeError):
        text = response.text or ""
        lines.append("Body:")
        snippet = text[:2000]
        lines.append(snippet)
        if len(text) > 2000:
            lines.append(f"... truncated {len(text) - 2000} characters ...")

    return "\n".join(lines)


@safe_tool
def elasticsearch_read(
    path: str = "",
    method: str = "GET",
    query: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    if not path:
        return "error: path is required"

    base_url = os.getenv("ELASTICSEARCH_URL", mc_service_url(9200))
    allowed_methods = {"GET", "HEAD", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"
    if not _is_read_only_path(path):
        return "error: path blocked to prevent write operations"
    if verb == "POST" and not _is_allowed_post_path(path):
        return "error: POST is limited to read-only endpoints such as *_search"

    try:
        timeout_seconds = max(1, min(int(timeout or 10), 30))
    except ValueError:
        return "error: timeout must be an integer"

    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    request_headers = {
        "Accept": "application/json",
    }
    if body:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    try:
        response = requests.request(
            verb,
            url,
            params=query,
            data=body,
            headers=request_headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return f"error: {exc}"

    lines = [
        f"URL: {response.url}",
        f"Status: {response.status_code} {response.reason}",
        f"Elapsed: {response.elapsed.total_seconds():.2f}s",
    ]

    try:
        payload = response.json()
        pretty = json.dumps(payload, indent=2)
        lines.append("JSON:")
        lines.append(pretty[:4000])
        if len(pretty) > 4000:
            lines.append(f"... truncated {len(pretty) - 4000} characters ...")
    except (ValueError, TypeError):
        text = response.text or ""
        lines.append("Body:")
        snippet = text[:2000]
        lines.append(snippet)
        if len(text) > 2000:
            lines.append(f"... truncated {len(text) - 2000} characters ...")

    return "\n".join(lines)


TOOL_SPECS = [
    ToolSpec(
        name="ping",
        description="Ping some host on the internet.",
        parameters={
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Hostname or IP address.",
                }
            },
            "required": ["host"],
        },
        handler=ping,
    ),
    ToolSpec(
        name="google_search",
        description="Search Google and return the top results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms to look up.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to fetch (1-10).",
                },
            },
            "required": ["query"],
        },
        handler=google_search,
    ),
    ToolSpec(
        name="http_request",
        description="Make a direct HTTP request (GET, HEAD, POST) to inspect status, headers, and body.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to request.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use.",
                    "enum": ["GET", "HEAD", "POST"],
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key/value pairs.",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "Request body when using POST.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30).",
                },
            },
            "required": ["url"],
        },
        handler=http_request,
    ),
    ToolSpec(
        name="grafana_request",
        description="Call the Grafana REST API using the configured service-account token.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Grafana API path, e.g. '/search' or '/dashboards/uid/abc123'.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use.",
                    "enum": ["GET", "HEAD", "POST", "PUT", "DELETE"],
                },
                "query": {
                    "type": "object",
                    "description": "Optional query parameters.",
                    "additionalProperties": {"type": "string"},
                },
                "headers": {
                    "type": "object",
                    "description": "Additional headers to include.",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "JSON body for POST/PUT requests.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30).",
                },
            },
            "required": ["path"],
        },
        handler=grafana_request,
    ),
    ToolSpec(
        name="kibana_request",
        description="Call the Kibana/Elasticsearch REST API (proxied via Kibana) with the configured credentials.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Kibana API path, e.g. '/api/saved_objects/_find'.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use (restricted to read-only verbs).",
                    "enum": ["GET", "HEAD", "POST"],
                },
                "query": {
                    "type": "object",
                    "description": "Optional query parameters.",
                    "additionalProperties": {"type": "string"},
                },
                "headers": {
                    "type": "object",
                    "description": "Additional headers to include.",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "JSON body for read-only POST requests (e.g. *_search).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30).",
                },
            },
            "required": ["path"],
        },
        handler=kibana_request,
    ),
    ToolSpec(
        name="elasticsearch_read",
        description="Call Elasticsearch directly with read-only safeguards (GET/HEAD, POST only for *_search).",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Elasticsearch API path, e.g. '/_cluster/health' or '/logs-*/_search'.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use.",
                    "enum": ["GET", "HEAD", "POST"],
                },
                "query": {
                    "type": "object",
                    "description": "Optional query parameters.",
                    "additionalProperties": {"type": "string"},
                },
                "headers": {
                    "type": "object",
                    "description": "Additional headers to include.",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "JSON body for search/count requests.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30).",
                },
            },
            "required": ["path"],
        },
        handler=elasticsearch_read,
    ),
    ToolSpec(
        name="logtrail_request",
        description="Call the Kibana Logtrail plugin API (default /logtrail) for log views.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Logtrail API path relative to LOGTRAIL_API_URL, defaults to 'default/json'.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method to use.",
                    "enum": ["GET", "POST"],
                },
                "query": {
                    "type": "object",
                    "description": "Optional query string parameters.",
                    "additionalProperties": {"type": "string"},
                },
                "headers": {
                    "type": "object",
                    "description": "Additional headers to include.",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "JSON body for POST requests (e.g., search payloads).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30).",
                },
            },
            "required": [],
        },
        handler=logtrail_request,
    ),
]

OPENAI_TOOLS = [spec.as_openai_tool() for spec in TOOL_SPECS]
GEMINI_TOOLS = [
    {
        "function_declarations": [spec.as_gemini_function() for spec in TOOL_SPECS],
    }
]
TOOL_HANDLERS = {spec.name: spec.handler for spec in TOOL_SPECS}


def needs_search(text: str) -> bool:
    lowered = text.lower()
    return "google" in lowered and ("search" in lowered or "live" in lowered)


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"error: unknown tool '{name}'"
    return handler(**arguments)


def tool_call(item) -> Dict[str, Any]:
    arguments = json.loads(item.arguments or "{}")
    result = execute_tool(item.name, arguments)
    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": result,
    }


class BaseBackend:
    def process(self, line: str) -> Tuple[str, Optional[str]]:
        raise NotImplementedError


class OpenAIBackend(BaseBackend):
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model
        self.context: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _request(self, force_tool: Optional[str] = None):
        kwargs = {"model": self.model, "tools": OPENAI_TOOLS, "input": self.context}
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        response = self.client.responses.create(**kwargs)
        token_tracker.record(response.usage)
        return response

    def _handle_tools(self, response) -> bool:
        changed = False
        for item in response.output:
            self.context.append(item)
            if item.type == "function_call":
                self.context.append(tool_call(item))
                changed = True
        return changed

    def process(self, line: str) -> Tuple[str, Optional[str]]:
        token_tracker.start_turn()
        self.context.append({"role": "user", "content": line})
        response = self._request("google_search" if needs_search(line) else None)
        while self._handle_tools(response):
            response = self._request()
        usage_report = token_tracker.end_turn()
        return response.output_text, usage_report


class GeminiBackend(BaseBackend):
    def __init__(self, model: str) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set for the Gemini backend.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS,
        )
        self.chat = self.model.start_chat(history=[])

    def _send_user_message(self, text: str, force_tool: Optional[str]) -> Any:
        tool_config = None
        if force_tool:
            tool_config = genai_types.ToolConfig(function_call={"name": force_tool})
        return self.chat.send_message(text, tool_config=tool_config)

    def _send_tool_response(self, name: str, output: str) -> Any:
        if not name:
            raise ValueError("Tool response missing function name.")
        payload = {
            "role": "tool",
            "parts": [
                {
                    "function_response": {
                        "name": name,
                        "response": {"output": output},
                    }
                }
            ],
        }
        return self.chat.send_message(payload)

    @staticmethod
    def _extract_function_calls(response: Any) -> List[Tuple[str, Dict[str, Any]]]:
        calls: List[Tuple[str, Dict[str, Any]]] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                name = getattr(function_call, "name", None) or getattr(function_call, "function_name", None)
                if not name and isinstance(function_call, dict):
                    name = function_call.get("name")
                if not name:
                    continue
                args_message = getattr(function_call, "args", None)
                arguments = message_to_dict(args_message)
                calls.append((name, arguments))
        return calls

    @staticmethod
    def _extract_text(response: Any) -> str:
        segments: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    segments.append(text)
        if not segments:
            text_attr = getattr(response, "text", None)
            if text_attr:
                segments.append(text_attr)
        return "\n".join(segments).strip()

    def process(self, line: str) -> Tuple[str, Optional[str]]:
        response = self._send_user_message(line, "google_search" if needs_search(line) else None)
        while True:
            calls = self._extract_function_calls(response)
            if not calls:
                break
            for name, arguments in calls:
                output = execute_tool(name, arguments)
                response = self._send_tool_response(name, output)
                break
        text = self._extract_text(response) or "(no response)"
        return text, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive agent that defaults to Google Gemini but can fall back to OpenAI."
    )
    parser.add_argument(
        "--backend",
        choices=("gemini", "openai"),
        default=os.getenv("AGENT_BACKEND", "gemini"),
        help="Choose which LLM backend to use (default: gemini).",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model to use when --backend=openai.",
    )
    parser.add_argument(
        "--target-mc",
        default=os.getenv("TARGET_MC", DEFAULT_TARGET_MC),
        help="Base MC URL used for Grafana/Kibana/Elasticsearch defaults.",
    )
    return parser.parse_args()


def create_backend(args: argparse.Namespace) -> BaseBackend:
    if args.backend == "openai":
        return OpenAIBackend(args.openai_model)
    return GeminiBackend(args.gemini_model)


def repl(backend: BaseBackend) -> None:
    while True:
        line = input("\033[1mAsk anything: \033[0m")
        response_text, usage_report = backend.process(line)
        print(f"%%; {response_text}\n")
        if usage_report:
            print(usage_report)
            print()


def main() -> None:
    args = parse_args()
    configure_target_mc(args.target_mc)
    backend = create_backend(args)
    try:
        repl(backend)
    except (EOFError, KeyboardInterrupt):
        print("Bye.")
        summary = token_tracker.session_summary()
        if summary:
            print(summary)


if __name__ == "__main__":
    main()
