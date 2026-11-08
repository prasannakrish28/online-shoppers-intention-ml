"""
Convenience script in the root directory that runs the full model training
and evaluation suite from `model/train_models.py`.
"""

import sys
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_script = os.path.join(current_dir, "model", "train_models.py")
    if os.path.exists(model_script):
        # Execute train_models.py in the model directory context
        with open(model_script, "r") as f:
            code = compile(f.read(), model_script, "exec")
            exec(code, {"__name__": "__main__", "__file__": model_script})
    else:
        print(f"Error: Could not locate {model_script}")
        sys.exit(1)

