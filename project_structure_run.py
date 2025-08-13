"""
Project Structure: Data Science 
This script creates the recommended folder structure for a data science project.
"""

import os

PROJECT_DIRS = [
    "data/raw",
    "data/processed",
    "notebooks/exploratory",
    "notebooks/feature_engineering",
    "notebooks/model",
    "src",
    "models",
    "reports",
]


def create_project_structure(base_path: str = "."):
    for dir_path in PROJECT_DIRS:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created: {full_path}")

    # Create placeholder .gitkeep files
    for dir_path in PROJECT_DIRS:
        full_path = os.path.join(base_path, dir_path, ".gitkeep")
        with open(full_path, "w") as f:
            f.write("")
        print(f"Added .gitkeep to: {full_path}")


if __name__ == "__main__":
    create_project_structure()
