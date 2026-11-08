"""`python -m app.mcp`: the MCP server over stdio.

stdout carries the protocol and nothing else: while it serves, the SDK points
the process's own stdout at stderr, so a stray print cannot break the
client's connection. The logs go to stderr in the API's format
(`configure_logging`): JSON when stderr is not a terminal, as under a client.
"""

from app.logging_config import configure_logging
from app.mcp.server import build_server


def main() -> None:
    configure_logging()
    build_server().run("stdio")


if __name__ == "__main__":
    main()
