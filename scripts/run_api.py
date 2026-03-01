"""Run the Quant Stack API server.

Usage:
    python -m scripts.run_api [--port 8000] [--host 0.0.0.0] [--reload]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Parse arguments and start the uvicorn server."""
    parser = argparse.ArgumentParser(description="Quant Stack API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run:\n"
            "  pip install 'quant-stack[api]'\n"
            "or:\n"
            "  pip install uvicorn[standard]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Starting Quant Stack API on http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
