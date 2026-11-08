"""
Helper script to execute the model training and evaluation pipeline from root.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "train_models.py")
    if os.path.exists(script_path):
        result = subprocess.run([sys.executable, script_path])
        sys.exit(result.returncode)
    else:
        print(f"Error: Could not find {script_path}")
        sys.exit(1)
