import argparse
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from agent_tools.clockify import ClockifyError, DEFAULT_BASE_URL, generate_week_from_history


INSTRUCTIONS = (
    "Tools to mirror a past week of Clockify time entries into a target week. "
    "Defaults to dry runs and uses the most recent past week as a template. "
    "Provide CLOCKIFY_API_KEY in the environment before connecting."
)

WRITE_ANNOTATIONS = ToolAnnotations(readOnlyHint=False, destructiveHint=False)


def build_server(host: str = "127.0.0.1", port: int = 8010, base_url: str = DEFAULT_BASE_URL) -> FastMCP:
    server = FastMCP(
        "clockify-auto-week",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
    )

    @server.tool(
        name="generate_clockify_week",
        description=(
            "Copy the most recent historical week of Clockify entries into a target week. "
            "Defaults to next week. Set dry_run=false to create entries."
        ),
        annotations=WRITE_ANNOTATIONS,
    )
    def generate_clockify_week(
        target_week_start: str = "",
        weeks_back: int = 2,
        dry_run: bool = True,
        allow_existing: bool = False,
        workspace_id: str = "",
        user_id: str = "",
    ) -> str:
        api_key = os.getenv("CLOCKIFY_API_KEY")
        if not api_key:
            return "error: CLOCKIFY_API_KEY is not set"
        try:
            return generate_week_from_history(
                api_key=api_key,
                base_url=base_url,
                target_week_start=target_week_start or None,
                weeks_back=weeks_back,
                dry_run=dry_run,
                allow_existing=allow_existing,
                workspace_id=workspace_id or None,
                user_id=user_id or None,
            )
        except ClockifyError as exc:
            return f"error: {exc}"

    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose a Clockify auto-fill helper over the Model Context Protocol.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="Transport for MCP (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE/HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port for SSE/HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Clockify API base URL (default: https://api.clockify.me/api/v1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_server(host=args.host, port=args.port, base_url=args.base_url)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
