from pathlib import Path

# Repo root = parent of the scripts/ folder
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
BIRDS = DATA / "birds"
CONTROLS = DATA / "controls"
META = DATA / "metadata"
FIGS = ROOT / "figures"
