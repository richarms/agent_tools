import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = ("You are a powerful assistant. Use all the tools to which you have access to answer queries about the MeerKAT telescope infrastructure system health and status. Your primary aim is to highlight anomolies and find errors.")

context = [{"role": "system", "content": SYSTEM_PROMPT}]

SESSION_STORE_DIR = os.path.join(tempfile.gettempdir(), "agent_tools_sessions")
SESSION_FILE_SUFFIX = ".json"

DEFAULT_MODEL = "gpt-5-mini"

ToolHandler = Callable[..., str]


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


def _json_safe_copy(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {key: _json_safe_copy(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_copy(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe_copy(model_dump())
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return _json_safe_copy(to_dict())
        except TypeError:
            pass
    if hasattr(value, "__dict__"):
        return _json_safe_copy(vars(value))
    return str(value)


def _strip_status_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_status_fields(val) for key, val in value.items() if key != "status"}
    if isinstance(value, list):
        return [_strip_status_fields(item) for item in value]
    return value


def _context_snapshot() -> list:
    return [_strip_status_fields(_json_safe_copy(item)) for item in context]


def _generate_session_id(payload: str) -> str:
    seed = f"{time.time_ns()}:{payload}".encode("utf-8", errors="ignore")
    return hashlib.sha256(seed).hexdigest()[:10]


def save_session_context() -> Optional[str]:
    directory = _ensure_session_store_dir()
    if not directory:
        return None
    serialized_context = _context_snapshot()
    payload = {
        "context": serialized_context,
        "saved_at": time.time(),
        "target_mc": {"scheme": _target_scheme, "host": _target_host},
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


def resume_session(session_id: Optional[str]) -> bool:
    path = _session_file_path(session_id or "")
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    saved_context = payload.get("context")
    if not isinstance(saved_context, list):
        return False
    saved_context = [_strip_status_fields(item) for item in saved_context]
    if not saved_context:
        saved_context = [{"role": "system", "content": SYSTEM_PROMPT}]
    else:
        first_item = saved_context[0]
        if not isinstance(first_item, dict) or first_item.get("role") != "system":
            saved_context = [{"role": "system", "content": SYSTEM_PROMPT}, *saved_context]
    context.clear()
    context.extend(saved_context)
    return True


DEFAULT_TARGET_MC = "http://lab-mc.sdp.kat.ac.za"
_DEFAULT_TARGET = urlparse(DEFAULT_TARGET_MC)
_DEFAULT_SCHEME = _DEFAULT_TARGET.scheme or "http"
_DEFAULT_HOST = _DEFAULT_TARGET.hostname or _DEFAULT_TARGET.netloc or "lab-mc.sdp.kat.ac.za"
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


_target_scheme, _target_host = _normalize_target_mc(DEFAULT_TARGET_MC)


def configure_target_mc(value: Optional[str]) -> None:
    global _target_scheme, _target_host
    _target_scheme, _target_host = _normalize_target_mc(value)


def configure_model(value: Optional[str]) -> None:
    global _model_name
    candidate = (value or "").strip()
    _model_name = candidate or DEFAULT_MODEL


def mc_service_url(port: int, suffix: str = "") -> str:
    host = _target_host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    normalized_suffix = suffix or ""
    if normalized_suffix and not normalized_suffix.startswith("/"):
        normalized_suffix = f"/{normalized_suffix}"
    return f"{_target_scheme}://{host}:{port}{normalized_suffix}"


def _coerce_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    candidate = value if value else default
    try:
        number = int(candidate)
    except (TypeError, ValueError):
        raise ValueError("invalid integer") from None
    return max(min_value, min(max_value, number))


def _format_response(
    response: requests.Response,
    *,
    include_headers: int = 0,
    include_body: bool = True,
    prefer_json: bool = False,
    json_limit: int = 4000,
    text_limit: int = 2000,
) -> str:
    lines = [
        f"URL: {response.url}",
        f"Status: {response.status_code} {response.reason}",
        f"Elapsed: {response.elapsed.total_seconds():.2f}s",
    ]

    if include_headers:
        header_items = list(response.headers.items())
        if header_items:
            lines.append("Headers:")
            for key, value in header_items[:include_headers]:
                lines.append(f"{key}: {value}")
            if len(header_items) > include_headers:
                omitted = len(header_items) - include_headers
                lines.append(f"... {omitted} more headers omitted ...")

    if not include_body:
        return "\n".join(lines)

    if prefer_json:
        try:
            payload = response.json()
            pretty = json.dumps(payload, indent=2)
            lines.append("JSON:")
            lines.append(pretty[:json_limit])
            if len(pretty) > json_limit:
                lines.append(f"... truncated {len(pretty) - json_limit} characters ...")
            return "\n".join(lines)
        except (ValueError, TypeError):
            pass

    text = response.text or ""
    lines.append("Body:")
    snippet = text[:text_limit]
    lines.append(snippet)
    if len(text) > text_limit:
        lines.append(f"... truncated {len(text) - text_limit} characters ...")
    return "\n".join(lines)


def _perform_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[str] = None,
    timeout_seconds: int = 10,
    allow_redirects: bool = True,
    include_headers: int = 0,
    include_body: bool = True,
    prefer_json: bool = False,
    json_limit: int = 4000,
    text_limit: int = 2000,
) -> str:
    try:
        response = requests.request(
            method,
            url,
            params=params,
            data=data,
            headers=headers,
            timeout=timeout_seconds,
            allow_redirects=allow_redirects,
        )
    except requests.RequestException as exc:
        return f"error: {exc}"
    return _format_response(
        response,
        include_headers=include_headers,
        include_body=include_body,
        prefer_json=prefer_json,
        json_limit=json_limit,
        text_limit=text_limit,
    )


def _resolve_kibana_auth_header() -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("KIBANA_API_KEY")
    basic_auth = os.getenv("KIBANA_BASIC_AUTH")
    allow_anonymous = os.getenv("KIBANA_ALLOW_NO_AUTH")
    allow_anonymous = (
        str(allow_anonymous).strip().lower() in {"1", "true", "yes", "on"}
        if allow_anonymous is not None
        else False
    )

    if api_key:
        return f"ApiKey {api_key}", None
    if basic_auth:
        encoded = base64.b64encode(basic_auth.encode("utf-8")).decode("utf-8")
        return f"Basic {encoded}", None
    if allow_anonymous:
        return None, None
    return None, "error: set KIBANA_API_KEY or KIBANA_BASIC_AUTH ('username:password')"


def _read_only_guard(path: str, verb: str) -> Optional[str]:
    if not _is_read_only_path(path):
        return "error: path blocked to prevent write operations"
    if verb == "POST" and not _is_allowed_post_path(path):
        return "error: POST is limited to read-only endpoints such as *_search"
    return None


configure_target_mc(os.getenv("TARGET_MC"))


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
_last_usage_report: Optional[str] = None


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
        timeout_seconds = _coerce_int(timeout, 10, 1, 30)
    except ValueError:
        return "error: timeout must be an integer"
    return _perform_request(
        verb,
        url,
        headers=headers,
        data=body,
        timeout_seconds=timeout_seconds,
        allow_redirects=True,
        include_headers=15,
        include_body=verb != "HEAD",
        text_limit=2000,
    )


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
        timeout_seconds = _coerce_int(timeout, 10, 1, 30)
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

    return _perform_request(
        verb,
        url,
        headers=request_headers,
        params=query,
        data=body,
        timeout_seconds=timeout_seconds,
        prefer_json=True,
        json_limit=4000,
        text_limit=2000,
    )


@safe_tool
def mesos_frameworks(
    path: str = "/master/frameworks",
    timeout: int = 10,
    limit: int = 5,
    include_completed: bool = False,
) -> str:
    normalized_path = "/" + (path or "/master/frameworks").lstrip("/")
    base_url = os.getenv("MESOS_MASTER_URL")
    if base_url:
        url = f"{base_url.rstrip('/')}{normalized_path}"
    else:
        url = mc_service_url(5050, normalized_path)
    try:
        timeout_seconds = _coerce_int(timeout, 10, 1, 30)
    except ValueError:
        return "error: timeout must be an integer"
    try:
        limit_count = _coerce_int(limit, 5, 1, 20)
    except ValueError:
        return "error: limit must be an integer"

    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return f"error: {exc}"

    if response.status_code >= 400:
        return f"error: {response.status_code} {response.reason}"

    try:
        payload = response.json()
    except ValueError:
        snippet = (response.text or "")[:1000]
        return f"error: expected JSON response, got: {snippet}"

    def summarize(frameworks, label):
        lines = [f"{label}: {len(frameworks)}"]
        for framework in frameworks[:limit_count]:
            name = framework.get("name", "unknown")
            framework_id = framework.get("id", "?")
            user = framework.get("user") or "n/a"
            hostname = framework.get("hostname") or ""
            roles = framework.get("roles")
            if not roles:
                role = framework.get("role")
                roles = [role] if role else []
            roles_text = ", ".join(sorted(str(role) for role in roles if role)) or "n/a"
            tasks = framework.get("tasks") or []
            resources = framework.get("resources") or {}
            cpus = resources.get("cpus")
            mem = resources.get("mem")
            gpus = resources.get("gpus")
            resource_bits = []
            if isinstance(cpus, (int, float)):
                resource_bits.append(f"cpus={cpus}")
            if isinstance(mem, (int, float)):
                resource_bits.append(f"mem={mem} MB")
            if isinstance(gpus, (int, float)):
                resource_bits.append(f"gpus={gpus}")
            resource_text = ", ".join(resource_bits) if resource_bits else "resources=n/a"
            host_text = f" host={hostname}" if hostname else ""
            lines.append(
                f"- {name} (id {framework_id}, user={user}{host_text}) "
                f"roles={roles_text} tasks={len(tasks)} {resource_text}"
            )
        if len(frameworks) > limit_count:
            lines.append(f"... {len(frameworks) - limit_count} more {label.lower()} omitted ...")
        return lines

    frameworks = payload.get("frameworks") or []
    completed = payload.get("completed_frameworks") or []

    sections = [f"URL: {url}"]
    sections.extend(summarize(frameworks, "Active frameworks"))
    if include_completed and completed:
        sections.append("")
        sections.extend(summarize(completed, "Completed frameworks"))
    elif include_completed:
        sections.append("Completed frameworks: 0")
    return "\n".join(sections)


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
    auth_header, auth_error = _resolve_kibana_auth_header()
    if auth_error:
        return auth_error

    base_url = os.getenv("KIBANA_API_URL", mc_service_url(5601))
    allowed_methods = {"GET", "HEAD", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"
    guard_error = _read_only_guard(path, verb)
    if guard_error:
        return guard_error

    try:
        timeout_seconds = _coerce_int(timeout, 10, 1, 30)
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

    return _perform_request(
        verb,
        url,
        headers=request_headers,
        params=query,
        data=body,
        timeout_seconds=timeout_seconds,
        prefer_json=True,
        json_limit=4000,
        text_limit=2000,
    )


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
    guard_error = _read_only_guard(path, verb)
    if guard_error:
        return guard_error

    try:
        timeout_seconds = _coerce_int(timeout, 10, 1, 30)
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

    return _perform_request(
        verb,
        url,
        headers=request_headers,
        params=query,
        data=body,
        timeout_seconds=timeout_seconds,
        prefer_json=True,
        json_limit=4000,
        text_limit=2000,
    )


@safe_tool
def logtrail_request(
    path: str = "default/json",
    method: str = "GET",
    query: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    auth_header, auth_error = _resolve_kibana_auth_header()
    if auth_error:
        return auth_error

    base_url = os.getenv("LOGTRAIL_API_URL", mc_service_url(5601, "/logtrail"))
    allowed_methods = {"GET", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"

    try:
        timeout_seconds = _coerce_int(timeout, 10, 1, 30)
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

    return _perform_request(
        verb,
        url,
        headers=request_headers,
        params=query,
        data=body,
        timeout_seconds=timeout_seconds,
        prefer_json=True,
        json_limit=4000,
        text_limit=2000,
    )


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
    ToolSpec(
        name="mesos_frameworks",
        description="Fetch framework information from the Mesos master (defaults to /master/frameworks).",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Custom Mesos API path (default '/master/frameworks').",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum frameworks to list per section (1-20).",
                },
                "include_completed": {
                    "type": "boolean",
                    "description": "When true, also summarize completed frameworks.",
                },
            },
            "required": [],
        },
        handler=mesos_frameworks,
    ),
]

tools = [spec.as_openai_tool() for spec in TOOL_SPECS]
TOOL_HANDLERS = {spec.name: spec.handler for spec in TOOL_SPECS}


def request_response(force_tool: Optional[str] = None):
    kwargs = {"model": _model_name, "tools": tools, "input": _context_snapshot()}
    if force_tool:
        kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
    response = client.responses.create(**kwargs)
    token_tracker.record(response.usage)
    return response


def tool_call(item):
    handler = TOOL_HANDLERS.get(item.name)
    if handler is None:
        result = f"error: unknown tool '{item.name}'"
    else:
        arguments = json.loads(item.arguments or "{}")
        result = handler(**arguments)
    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": result,
    }


def handle_tools(response) -> bool:
    changed = False
    for item in response.output:
        context.append(item)
        if item.type == "function_call":
            context.append(tool_call(item))
            changed = True
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive agent REPL.")
    parser.add_argument(
        "--target-mc",
        default=os.getenv("TARGET_MC", DEFAULT_TARGET_MC),
        help="Base MC URL used for Grafana/Kibana/Elasticsearch defaults.",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a saved session using the hashed reference printed on exit.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI Responses API model to use (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def needs_search(text: str) -> bool:
    lowered = text.lower()
    return "google" in lowered and ("search" in lowered or "live" in lowered)


def process(line: str) -> str:
    global _last_usage_report
    token_tracker.start_turn()
    context.append({"role": "user", "content": line})
    response = request_response("google_search" if needs_search(line) else None)
    while handle_tools(response):
        response = request_response()
    _last_usage_report = token_tracker.end_turn()
    return response.output_text


def latest_usage_report(clear: bool = True) -> Optional[str]:
    global _last_usage_report
    report = _last_usage_report
    if clear:
        _last_usage_report = None
    return report


def repl() -> None:
    while True:
        line = input("\033[1mAsk anything: \033[0m")
        result = process(line)
        print(f"%%; {result}\n")
        report = latest_usage_report()
        if report:
            print(report)
            print()


def main() -> None:
    args = parse_args()
    configure_target_mc(args.target_mc)
    configure_model(args.model)
    if args.resume:
        if resume_session(args.resume):
            print(f"Resumed session '{args.resume}'.")
        else:
            print(f"Could not find session '{args.resume}', starting a new one.")
    try:
        repl()
    except (EOFError, KeyboardInterrupt):
        print("Bye.")
    finally:
        saved_id = save_session_context()
        if saved_id:
            print(f"Session context saved. Reference: {saved_id}")
            print(f"Resume later with: uv run agent_tools.py --resume {saved_id}")
        else:
            print("Unable to save session context (temporary storage unavailable).")


if __name__ == "__main__":
    main()
