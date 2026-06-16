"""Make the project importable as ``src.*`` regardless of pytest's rootdir."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
