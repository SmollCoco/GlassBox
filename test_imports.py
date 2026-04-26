import sys
from pathlib import Path

base_dir = Path(r"c:\Users\Master\Desktop\AI_Workspace\Project\GlassBox\packages")
for p in base_dir.iterdir():
    if p.is_dir() and (p / "src").exists():
        sys.path.insert(0, str(p / "src"))

import traceback

modules_to_test = [
    "GlassBox.numpandas",
    "GlassBox.eda",
    "GlassBox.preprocessing",
    "GlassBox.split",
    "GlassBox.pipeline",
    "GlassBox.optimization",
    "GlassBox.ml",
    "GlassBox.benchmark",
]

for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"SUCCESS: {mod}")
    except Exception as e:
        print(f"FAILED: {mod}")
        traceback.print_exc()

