import sys
from pathlib import Path

# The modules in src/ import each other by plain name (as when running
# src/main.py), so put src/ on the path for the tests too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
