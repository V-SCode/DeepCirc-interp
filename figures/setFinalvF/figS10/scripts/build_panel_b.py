"""Rebuild S10b from the unchanged upper-decile distribution source."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from figures.setFinalvF._shared.preserved_distribution_panels import build_distribution_panel

if __name__ == "__main__":
    build_distribution_panel("c", "b")
