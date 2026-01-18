from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from . import utils


def _find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find the project root by walking upward until we see a marker file.
    Works in notebooks, scripts, and when installed editable.
    """
    markers = {"pyproject.toml", "README.md"}
    cur = (start or Path.cwd()).resolve()

    for p in [cur, *cur.parents]:
        if any((p / m).exists() for m in markers):
            return p

    # Fall back: package location (works after install)
    # src/gkr/_env.py -> parents[2] should be project root in editable layout
    return Path(__file__).resolve().parents[2]


def init(*, verbose: bool = False, root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Initialize environment helpers for notebooks / demos.

    Returns a dict with paths + utility functions, without sys.path hacks.
    """
    project_root = Path(root).resolve() if root else _find_project_root()
    scripts_path = project_root / "src" / "gkr"
    notebooks_path = project_root / "notebooks"

    env: Dict[str, Any] = {
        "PROJECT_PATH": project_root,
        "SCRIPTS_PATH": scripts_path,
        "NOTEBOOKS_PATH": notebooks_path,

        # helpers
        "print_header": utils.print_header,
        "get_directory_tree": utils.get_directory_tree,
        "get_subdirectories": utils.get_subdirectories,
        "display_aligned": utils.display_aligned,

        # colors
        "RED": utils.RED,
        "GREEN": utils.GREEN,
        "YELLOW": utils.YELLOW,
        "PINK": utils.PINK,
        "BLUE": utils.BLUE,
        "PURPLE": utils.PURPLE,
        "RESET": utils.RESET,
    }

    # A "path" dict like you had before (first-level subdirs of project root)
    env["path"] = utils.get_subdirectories(project_root, depth=0)

    if verbose:
        utils.print_header("Project directory tree")
        utils.print_header(f"Base path: {project_root}", level=2)
        utils.get_directory_tree(project_root, base_name="project", print_paths=True)

        utils.print_header("Paths to first-level subdirectories stored in 'path' dictionary", level=2)
        for name, p in env["path"].items():
            print(f"├─ path['{name}'] = {p}")

    return env
