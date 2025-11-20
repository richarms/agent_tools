import base64
from typing import Any

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


def _coerce_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    candidate = value if value else default
    try:
        number = int(candidate)
    except (TypeError, ValueError):
        raise ValueError("invalid integer") from None
    return max(min_value, min(max_value, number))


def format_number(value: int) -> str:
    return f"{value:,}"
