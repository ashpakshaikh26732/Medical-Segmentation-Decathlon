# setup_path.py

import sys
import os

def add_project_root():
    """
    Adds the absolute path to the Project folder.
    Hardcoded, simple, and works everywhere.
    """
    # project_path = "/content/medical-segmentation-decathlon/Project"  # ← Colab path
    # Or use your local path if you're in VS Code:
    project_path = "/home/levi/medical-segmentation-decathlon/Project"

    if project_path not in sys.path:
        sys.path.insert(0, project_path)
