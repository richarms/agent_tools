import os
import subprocess
import random
import json
from typing import Callable, Dict
import requests
from openai import OpenAI
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks for up-to-date information beyond your training date,"
    "or explicitly requests a Google search, you MUST call the `google_search` tool "
    "before responding. Use the search results to craft your answer. If the tool fails, "
    "explain the failure and provide next steps. If the user is only interested in "
    "whether a website is up, preferentially use the ping tool."
)

context = [{"role": "system", "content": SYSTEM_PROMPT}]

tools = [{
   "type": "function", "name": "ping",
   "description": "ping some host on the internet",
   "parameters": {
       "type": "object", "properties": {
           "host": {
             "type": "string", "description": "hostname or IP",
            },
       },
       "required": ["host"],
    },},
{
   "type": "function", "name": "google_search",
   "description": "Search Google and return the top results.",
   "parameters": {
       "type": "object", "properties": {
           "query": {
             "type": "string", "description": "Search terms to look up.",
           },
           "num_results": {
             "type": "integer",
             "description": "Number of results to fetch (1-10).",
           },
       },
       "required": ["query"],
    },}]

def ping(host=""):
    try:
        result = subprocess.run(
            ["ping", "-c", "5", host],
            text=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE)
        return result.stdout
    except Exception as e:
        return f"error: {e}"

def google_search(query="", num_results=3):
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cx:
        return "error: GOOGLE_API_KEY and GOOGLE_CSE_ID must be set"
    num = max(1, min(int(num_results or 3), 10))
    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx, "q": query, "num": num},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return "no results"
        lines = []
        for item in items:
            title = item.get("title", "untitled")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"{title}\n{link}\n{snippet}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"error: {e}"

def call(tools, force_tool=None):        # now takes an arg
    kwargs = {"model": "gpt-5-mini", "tools": tools, "input": context}
    if force_tool:
        kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
    return client.responses.create(**kwargs)

ToolHandler = Callable[..., str]
TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "ping": ping,
    "google_search": google_search,
}

def tool_call(item):    # just handles one tool
    handler = TOOL_HANDLERS.get(item.name)
    if handler is None:
        result = f"error: unknown tool '{item.name}'"
    else:
        args = json.loads(item.arguments or "{}")
        result = handler(**args)
    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": result
    }

def handle_tools(tools, response):
    changed = False
    new_items = []
    for item in response.output:
        new_items.append(item)
        if item.type == "function_call":
            new_items.append(tool_call(item))
            changed = True
    if new_items:
        context.extend(new_items)
    return changed

def process(line):
    context.append({"role": "user", "content": line})
    lower_line = line.lower()
    force_tool = None
    if "google" in lower_line and ("search" in lower_line or "live" in lower_line):
        force_tool = "google_search"
    response = call(tools, force_tool=force_tool)
    # new code: resolve tool calls
    while handle_tools(tools, response):
        response = call(tools)        
    return response.output_text

def main():
    try:
        while True:
            line = input("$$; ")
            result = process(line)
            print(f"%%; {result}\n")
    except KeyboardInterrupt:
        print("Here's the final context string:")
        print(context[1:])

if __name__ == "__main__": main()
