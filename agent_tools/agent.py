import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple

from openai import OpenAI
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.protobuf.json_format import MessageToDict

from agent_tools import config
from agent_tools import tools
from agent_tools import utils


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


def format_usage(label: str, totals: UsageTotals) -> str:
    parts = [f"{label}: total={utils.format_number(totals.total_tokens)}"]
    if totals.input_tokens or totals.cached_input_tokens:
        segment = f"input={utils.format_number(totals.input_tokens)}"
        if totals.cached_input_tokens:
            segment += f" (+ {utils.format_number(totals.cached_input_tokens)} cached)"
        parts.append(segment)
    if totals.output_tokens:
        segment = f"output={utils.format_number(totals.output_tokens)}"
        if totals.reasoning_tokens:
            segment += f" (reasoning {utils.format_number(totals.reasoning_tokens)})"
        parts.append(segment)
    return " ".join(parts)


token_tracker = TokenTracker()
_last_usage_report: Optional[str] = None


def needs_search(text: str) -> bool:
    lowered = text.lower()
    return "google" in lowered and ("search" in lowered or "live" in lowered)


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

def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    handler = tools.TOOL_HANDLERS.get(name)
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

    @property
    def context(self) -> List[Dict[str, Any]]:
        return []


class OpenAIBackend(BaseBackend):
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model
        self._context: List[Dict[str, Any]] = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    @property
    def context(self) -> List[Dict[str, Any]]:
        return self._context
    
    def set_context(self, context: List[Dict[str, Any]]) -> None:
        self._context = context

    def _context_snapshot(self) -> list:
        return [utils._strip_status_fields(utils._json_safe_copy(item)) for item in self._context]

    def _request(self, force_tool: Optional[str] = None):
        kwargs = {
            "model": self.model,
            "tools": tools.OPENAI_TOOLS,
            "input": self._context_snapshot()
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        response = self.client.responses.create(**kwargs)
        token_tracker.record(response.usage)
        return response

    def _handle_tools(self, response) -> bool:
        changed = False
        for item in response.output:
            self._context.append(item)
            if item.type == "function_call":
                self._context.append(tool_call(item))
                changed = True
        return changed

    def process(self, line: str) -> Tuple[str, Optional[str]]:
        token_tracker.start_turn()
        self._context.append({"role": "user", "content": line})
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
            system_instruction=config.SYSTEM_PROMPT,
            tools=tools.GEMINI_TOOLS,
        )
        self.chat = self.model.start_chat(history=[])
        # We maintain a shadow context for saving, though full fidelity is hard
        self._shadow_context: List[Dict[str, Any]] = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    @property
    def context(self) -> List[Dict[str, Any]]:
        return self._shadow_context
    
    def set_context(self, context: List[Dict[str, Any]]) -> None:
        self._shadow_context = context
        # Note: We cannot easily restore Gemini chat state from this list without 
        # extensive conversion logic. For now, we just accept that resuming a session 
        # into Gemini might start a fresh chat with system prompt, or we would need
        # to implement 'history' reconstruction.
        # A simple reconstruction attempt could be done here if needed.

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
        # Update shadow context
        self._shadow_context.append({"role": "user", "content": line})
        
        response = self._send_user_message(line, "google_search" if needs_search(line) else None)
        while True:
            calls = self._extract_function_calls(response)
            if not calls:
                break
            for name, arguments in calls:
                # Shadow context update (approximate)
                self._shadow_context.append({
                    "role": "assistant", 
                    "function_call": {"name": name, "arguments": json.dumps(arguments)}
                })
                
                output = execute_tool(name, arguments)
                
                self._shadow_context.append({
                    "role": "tool", 
                    "name": name, 
                    "content": output
                })
                
                response = self._send_tool_response(name, output)
                break
        
        text = self._extract_text(response) or "(no response)"
        self._shadow_context.append({"role": "assistant", "content": text})
        
        # Gemini API currently doesn't return token usage in the same format/easily in all SDK versions
        # so we return None for usage report for now.
        return text, None


def create_backend(name: str, gemini_model: str, openai_model: str) -> BaseBackend:
    if name == "openai":
        return OpenAIBackend(openai_model)
    elif name == "gemini":
        return GeminiBackend(gemini_model)
    else:
        # Default to Gemini if unknown, or raise error. 
        # Given the previous logic, let's default to Gemini if "gemini" or anything else
        # but explicitly check for openai.
        return GeminiBackend(gemini_model)


# Singleton backend holder for the CLI to access if needed, 
# but CLI should prefer passing the backend instance.
# We removed the global 'context' variable in favor of the backend's context.