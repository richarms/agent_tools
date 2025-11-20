import base64
import json
import os
import subprocess
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, List, Tuple

import requests

from agent_tools import config
from agent_tools import utils

ToolHandler = Callable[..., str]


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
        timeout_seconds = utils._coerce_int(timeout, 10, 1, 30)
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

    base_url = os.getenv("GRAFANA_API_URL", config.mc_service_url(3000, "/api"))
    allowed_methods = {"GET", "HEAD", "POST", "PUT", "DELETE"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"

    try:
        timeout_seconds = utils._coerce_int(timeout, 10, 1, 30)
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
        url = config.mc_service_url(5050, normalized_path)
    try:
        timeout_seconds = utils._coerce_int(timeout, 10, 1, 30)
    except ValueError:
        return "error: timeout must be an integer"
    try:
        limit_count = utils._coerce_int(limit, 5, 1, 20)
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


def _read_only_guard(path: str, verb: str) -> Optional[str]:
    if not _is_read_only_path(path):
        return "error: path blocked to prevent write operations"
    if verb == "POST" and not _is_allowed_post_path(path):
        return "error: POST is limited to read-only endpoints such as *_search"
    return None


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

    base_url = os.getenv("KIBANA_API_URL", config.mc_service_url(5601))
    allowed_methods = {"GET", "HEAD", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"
    guard_error = _read_only_guard(path, verb)
    if guard_error:
        return guard_error

    try:
        timeout_seconds = utils._coerce_int(timeout, 10, 1, 30)
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

    base_url = os.getenv("ELASTICSEARCH_URL", config.mc_service_url(9200))
    allowed_methods = {"GET", "HEAD", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"
    guard_error = _read_only_guard(path, verb)
    if guard_error:
        return guard_error

    try:
        timeout_seconds = utils._coerce_int(timeout, 10, 1, 30)
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

    base_url = os.getenv("LOGTRAIL_API_URL", config.mc_service_url(5601, "/logtrail"))
    allowed_methods = {"GET", "POST"}
    verb = (method or "GET").upper()
    if verb not in allowed_methods:
        return f"error: unsupported method '{verb}', use one of {sorted(allowed_methods)}"

    try:
        timeout_seconds = utils._coerce_int(timeout, 10, 1, 30)
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

OPENAI_TOOLS = [spec.as_openai_tool() for spec in TOOL_SPECS]
GEMINI_TOOLS = [
    {
        "function_declarations": [spec.as_gemini_function() for spec in TOOL_SPECS],
    }
]
TOOL_HANDLERS = {spec.name: spec.handler for spec in TOOL_SPECS}