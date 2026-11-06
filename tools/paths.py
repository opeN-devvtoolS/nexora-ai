# tools/paths.py
# Safety: the agent can only touch the "workspace" folder.
# Every other tool file uses get_path() from here to build file paths.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = PROJECT_ROOT / "workspace"

ALLOWED_FOLDERS = ["input", "knowledge", "output"]


def get_path(folder, filename):

    if folder not in ALLOWED_FOLDERS:
        raise ValueError("Invalid folder")

    folder_path = (WORKSPACE / folder).resolve()
    file_path = (folder_path / filename).resolve()

    try:
        file_path.relative_to(folder_path)
    except ValueError:
        raise ValueError("Access outside workspace is not allowed")

    return file_path
