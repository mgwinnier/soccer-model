"""Verify that project-local credentials are wired up.  `python -m src.data.secrets_check`"""
from __future__ import annotations

from ..config import load_secrets, PROJECT_ROOT


def main() -> None:
    status = load_secrets()
    print("Credential check (project-local):\n")

    af = status["api_football"]
    print(f"  API-Football key : {'FOUND ✓' if af else 'missing'}"
          f"   (set API_FOOTBALL_KEY in .env or secrets/.env)")
    if af:
        from .lineups import connectivity_check
        verdict = connectivity_check()
        if verdict == "ok":
            print("      reachable ✓ — lineups + injuries will activate")
        elif verdict == "blocked_by_network":
            print("      ⚠ key is set but the API is BLOCKED BY YOUR NETWORK")
            print("        (a Ubiquiti/UniFi firewall is filtering api-sports.io).")
            print("        Allowlist 'v3.football.api-sports.io' / '*.api-sports.io'")
            print("        in your UniFi content filter, or use another network.")
        else:
            print(f"      ⚠ key set but not reachable: {verdict}")

    kg = status["kaggle"]
    print(f"  Kaggle token     : {'FOUND ✓' if kg else 'missing'}"
          f"   (KAGGLE_USERNAME + KAGGLE_KEY in .env, or secrets/kaggle.json)")
    if status.get("kaggle_key_no_user"):
        print("      ⚠ Found a Kaggle KEY/token but no KAGGLE_USERNAME — the Kaggle")
        print("        CLI needs both. Add KAGGLE_USERNAME=<your kaggle account name>.")

    print("\nWhat each unlocks:")
    print("  • API-Football → confirmed XI + injuries (sharpens imminent matches)")
    print("  • Kaggle       → FIFA player ratings (turns 'who's out' into a")
    print("                   strength penalty; also squad-strength features)")
    if af and kg:
        print("\nBoth present — run `python -m src.data.download` then reload the")
        print("dashboard; injury-adjusted predictions will activate for upcoming games.")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
