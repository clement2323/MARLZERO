"""Launch the Reversi FastAPI server."""
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "uvicorn",
        "morris_rl.inference.reversi_server:app",
        "--host", "0.0.0.0",
        "--port", "8001",
        "--reload",
    ],
    check=True,
)
