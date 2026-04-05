"""CLI entry point for trading persona management.

Usage::

    py -3 -m scripts.run_persona --list
    py -3 -m scripts.run_persona --activate crisis
    py -3 -m scripts.run_persona --deactivate crisis
    py -3 -m scripts.run_persona --pause core
    py -3 -m scripts.run_persona --brief core
    py -3 -m scripts.run_persona --brief-all
    py -3 -m scripts.run_persona --check-events
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Parse arguments and run persona commands."""
    parser = argparse.ArgumentParser(
        description="Quant Stack trading persona manager"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all personas and their states"
    )
    parser.add_argument(
        "--activate", metavar="ID", help="Activate a persona by ID"
    )
    parser.add_argument(
        "--deactivate", metavar="ID", help="Deactivate a persona by ID"
    )
    parser.add_argument(
        "--pause", metavar="ID", help="Pause an active persona by ID"
    )
    parser.add_argument(
        "--brief", metavar="ID", help="Generate a research brief for a persona"
    )
    parser.add_argument(
        "--brief-all",
        action="store_true",
        help="Generate briefs for all active personas",
    )
    parser.add_argument(
        "--check-events",
        action="store_true",
        help="Run event detection for dormant personas",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Total capital for budget sizing (default: 100000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config/settings.yaml)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # Check personas are enabled
    if not config.get("personas", {}).get("enabled", False):
        print("Personas are not enabled in config. Set personas.enabled: true")
        sys.exit(1)

    from src.services.persona_service import PersonaService
    svc = PersonaService(config)

    if args.list:
        _cmd_list(svc)
    elif args.activate:
        _cmd_activate(svc, args.activate)
    elif args.deactivate:
        _cmd_deactivate(svc, args.deactivate)
    elif args.pause:
        _cmd_pause(svc, args.pause)
    elif args.brief:
        _cmd_brief(svc, args.brief, args.capital)
    elif args.brief_all:
        _cmd_brief_all(svc, args.capital)
    elif args.check_events:
        _cmd_check_events(svc)
    else:
        parser.print_help()


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

_STATE_COLOURS = {
    "active": "\033[92m",    # green
    "pending": "\033[93m",   # amber
    "dormant": "\033[90m",   # grey
    "paused": "\033[94m",    # blue
}
_RESET = "\033[0m"


def _cmd_list(svc: "PersonaService") -> None:
    """List all personas with their states."""
    personas = svc.get_all_personas()
    if not personas:
        print("No personas configured.")
        return

    print(f"\n{'ID':<16} {'Name':<22} {'Type':<16} {'State':<12} {'Capital':<10} {'Strategy':<16}")
    print("-" * 94)
    for p in personas:
        state = p.get("state", "unknown")
        colour = _STATE_COLOURS.get(state, "")
        print(
            f"{p['persona_id']:<16} "
            f"{p['name']:<22} "
            f"{p['type']:<16} "
            f"{colour}{state:<12}{_RESET} "
            f"{p['capital_pct']:.0%}{'':>6} "
            f"{p['strategy']:<16}"
        )
    print()


def _cmd_activate(svc: "PersonaService", persona_id: str) -> None:
    """Activate a persona."""
    result = svc.activate_persona(persona_id)
    print(result["message"])


def _cmd_deactivate(svc: "PersonaService", persona_id: str) -> None:
    """Deactivate a persona."""
    result = svc.deactivate_persona(persona_id)
    print(result["message"])


def _cmd_pause(svc: "PersonaService", persona_id: str) -> None:
    """Pause a persona."""
    result = svc.pause_persona(persona_id)
    print(result["message"])


def _cmd_brief(
    svc: "PersonaService", persona_id: str, capital: float
) -> None:
    """Generate a research brief."""
    print(f"Generating brief for '{persona_id}'...")
    brief = svc.generate_brief_now(persona_id, total_capital=capital)
    if brief is None:
        print(f"Failed to generate brief for '{persona_id}'")
        return

    print(f"\n{'='*60}")
    print(brief.get("narrative", "No narrative generated."))
    print(f"{'='*60}")

    recs = brief.get("recommendations", [])
    if recs:
        print(f"\nTrade Recommendations ({len(recs)}):")
        for r in recs:
            print(
                f"  {r['direction']} {r['ticker']}: "
                f"{r['suggested_quantity']} shares @ {r['entry_price']:.2f} "
                f"(~GBP {r['suggested_size_gbp']:,.0f})"
            )
    else:
        print("\nNo trade recommendations at this time.")


def _cmd_brief_all(svc: "PersonaService", capital: float) -> None:
    """Generate briefs for all active personas."""
    personas = svc.get_all_personas()
    active = [p for p in personas if p.get("state") == "active"]
    if not active:
        print("No active personas.")
        return

    for p in active:
        pid = p["persona_id"]
        print(f"\n--- {p['name']} ({pid}) ---")
        _cmd_brief(svc, pid, capital)


def _cmd_check_events(svc: "PersonaService") -> None:
    """Run event detection."""
    print("Checking event conditions for dormant personas...")
    newly_pending = svc.check_events()
    if not newly_pending:
        print("No events detected. All dormant personas remain dormant.")
        return

    print(f"\n{len(newly_pending)} persona(s) moved to PENDING:")
    for entry in newly_pending:
        print(f"  {entry['name']} ({entry['persona_id']}): {entry['reason']}")
    print("\nUse --activate <id> to confirm activation.")


if __name__ == "__main__":
    main()
