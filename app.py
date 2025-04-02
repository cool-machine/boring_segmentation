import os
import subprocess
import sys

if __name__ == "__main__":
    port = os.environ.get("PORT", "8501")
    subprocess.call([
        "streamlit", "run", "streamlit_run.py",  # Note the path
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false"
    ])