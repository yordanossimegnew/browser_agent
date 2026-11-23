"""Entry point. The project is a FastAPI application; a terminal CLI is included for local use.

    uvicorn main:app --reload                                  # API server (docs at /docs)
    python main.py                                             # same, on port 8000
    python main.py cli "Compare the free tiers of Vercel and Netlify"
    python main.py cli --show "..."                            # watch the browser window
"""

from __future__ import annotations

import argparse
import sys

from src.api import create_app
from src.utils import load_settings, setup_logging

settings = load_settings()
setup_logging(settings.log_level, settings.log_dir)
app = create_app(settings)

DIM, BOLD, CYAN, RED, GREEN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[31m", "\033[32m", "\033[0m"


def print_event(event: dict) -> None:
    kind = event["type"]
    if kind == "thought" and not event.get("final"):
        print(f"\n{event['text']}")
    elif kind == "action":
        print(f"{CYAN}> {event['tool']}{RESET} {DIM}{event['args']}{RESET}")
    elif kind == "observation":
        color = RED if event["is_error"] else DIM
        print(f"{color}  {event.get('message') or event['url']}{RESET}")
    elif kind == "error":
        print(f"{RED}{event['message']}{RESET}")


def cli(args: argparse.Namespace) -> None:
    from src.api.runs import RunManager
    from src.agent import open_checkpointer

    cfg = load_settings(headless=False if args.show else None)
    manager = RunManager(cfg, open_checkpointer(cfg.sessions_dir))
    run = manager.start(args.task)
    print(f"{DIM}run {run.id} · model {cfg.model} · search {cfg.search_url.split('/')[2]}{RESET}")
    for event in manager.stream(run):
        if event:
            print_event(event)
    result = run.result or {}
    print(f"\n{BOLD}{GREEN}[{result.get('status')}]{RESET}\n\n{result.get('final_text', '')}")
    if result.get("status") == "error":
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser Agent")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Run the FastAPI server (default).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("cli", help="Run one research task in the terminal.")
    c.add_argument("task")
    c.add_argument("--show", action="store_true", help="Show the browser window (not headless).")

    args = parser.parse_args()
    if args.command == "cli":
        cli(args)
    else:
        import uvicorn

        uvicorn.run(app, host=getattr(args, "host", "0.0.0.0"), port=getattr(args, "port", 8000))


if __name__ == "__main__":
    main()
