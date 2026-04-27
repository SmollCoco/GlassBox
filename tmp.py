# scripts/screenshot3_nemoclaw_docker.py
import subprocess
import sys

cmd = [
    "docker", "run", "--rm",
    "-v", "./test_data:/data",
    "-v", "./results:/results",
    "glassbox-env:latest",
    "python", "-m", "GlassBox.autofit.cli",
    "--data", "/data/test_model.csv",
    "--target", "target",
    "--output", "/results/best_model.pkl",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, text=True, capture_output=True)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

sys.exit(result.returncode)