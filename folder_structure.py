import os

def print_directory_structure(startpath, exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = set()
    if exclude_files is None:
        exclude_files = set()

    for root, dirs, files in os.walk(startpath):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)

        # Exclude specified files
        for f in files:
            if f not in exclude_files:
                print(f"{subindent}{f}")

# Replace with your project's root directory
project_root = "C:/Users/Kevin/Documents/ScenGen"

# Specify directories and files to exclude
exclude_dirs = {"__pycache__", ".git", ".vscode", "venv", ".env"}
exclude_files = {".DS_Store", ".gitignore", ".env"}

print_directory_structure(project_root, exclude_dirs, exclude_files)