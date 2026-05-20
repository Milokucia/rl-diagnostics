"""
rl-diagnostics CLI entry point.

Usage:
    rl-diagnostics                        # start server on default port 7842
    rl-diagnostics --port 8000
    rl-diagnostics --host 0.0.0.0         # expose on network
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="rl-diagnostics",
        description="AI-powered RL training diagnostics server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7842, help="Port to bind (default: 7842)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    from rl_diagnostics.server import create_app
    app = create_app()

    print(f"rl-diagnostics running at http://{args.host}:{args.port}")
    print(f"Supported algorithms: ppo, sac, td3, ddpg, dqn, custom")
    print(f"Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
