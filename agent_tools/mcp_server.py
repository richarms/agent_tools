import argparse
import os
from typing import Dict, Optional, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from agent_tools import config, tools


READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)


def build_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """
    Create a FastMCP server that exposes the existing agent_tools functions as MCP tools.
    """
    server = FastMCP(
        "agent-tools",
        instructions=config.SYSTEM_PROMPT,
        host=host,
        port=port,
    )

    @server.tool(
        name="ping",
        description="Ping some host on the internet.",
        annotations=READ_ONLY,
    )
    def ping_tool(host: str) -> str:
        return tools.ping(host=host)

    @server.tool(
        name="google_search",
        description="Search Google and return the top results.",
        annotations=READ_ONLY,
    )
    def google_search_tool(query: str, num_results: int = 3) -> str:
        return tools.google_search(query=query, num_results=num_results)

    @server.tool(
        name="http_request",
        description="Make a direct HTTP request (GET, HEAD, POST) to inspect status, headers, and body.",
        annotations=READ_ONLY,
    )
    def http_request_tool(
        url: str,
        method: Literal["GET", "HEAD", "POST"] = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 10,
    ) -> str:
        return tools.http_request(
            url=url,
            method=method,
            headers=headers,
            body=body,
            timeout=timeout,
        )

    @server.tool(
        name="grafana_request",
        description="Call the Grafana REST API using the configured service-account token.",
        annotations=READ_ONLY,
    )
    def grafana_request_tool(
        path: str,
        method: Literal["GET", "HEAD", "POST"] = "GET",
        query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 10,
    ) -> str:
        return tools.grafana_request(
            path=path,
            method=method,
            query=query,
            headers=headers,
            body=body,
            timeout=timeout,
        )

    @server.tool(
        name="kibana_request",
        description="Call the Kibana/Elasticsearch REST API (proxied via Kibana) with the configured credentials.",
        annotations=READ_ONLY,
    )
    def kibana_request_tool(
        path: str,
        method: Literal["GET", "HEAD", "POST", "PUT", "DELETE"] = "GET",
        query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 10,
    ) -> str:
        return tools.kibana_request(
            path=path,
            method=method,
            query=query,
            headers=headers,
            body=body,
            timeout=timeout,
        )

    @server.tool(
        name="elasticsearch_read",
        description="Call Elasticsearch directly with read-only safeguards (GET/HEAD, POST only for *_search).",
        annotations=READ_ONLY,
    )
    def elasticsearch_read_tool(
        path: str,
        method: Literal["GET", "HEAD", "POST"] = "GET",
        query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 10,
    ) -> str:
        return tools.elasticsearch_read(
            path=path,
            method=method,
            query=query,
            headers=headers,
            body=body,
            timeout=timeout,
        )

    @server.tool(
        name="logtrail_request",
        description="Call the Kibana Logtrail plugin API (default /logtrail) for log views.",
        annotations=READ_ONLY,
    )
    def logtrail_request_tool(
        path: str = "default/json",
        method: Literal["GET", "POST"] = "GET",
        query: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 10,
    ) -> str:
        return tools.logtrail_request(
            path=path,
            method=method,
            query=query,
            headers=headers,
            body=body,
            timeout=timeout,
        )

    @server.tool(
        name="mesos_frameworks",
        description="Fetch framework information from the Mesos master (defaults to /master/frameworks).",
        annotations=READ_ONLY,
    )
    def mesos_frameworks_tool(
        path: str = "/master/frameworks",
        timeout: int = 10,
        limit: int = 5,
        include_completed: bool = False,
    ) -> str:
        return tools.mesos_frameworks(
            path=path,
            timeout=timeout,
            limit=limit,
            include_completed=include_completed,
        )

    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose agent_tools over the Model Context Protocol.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="Transport for MCP (default: stdio).",
    )
    parser.add_argument(
        "--target-mc",
        default=os.getenv("TARGET_MC", config.DEFAULT_TARGET_MC),
        help="Base MC URL to use when building Grafana/Kibana/Elasticsearch defaults.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE/HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE/HTTP transports (ignored for stdio).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.configure_target_mc(args.target_mc)
    server = build_server(host=args.host, port=args.port)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
