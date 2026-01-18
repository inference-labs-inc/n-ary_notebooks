from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


def get_subdirectories(
    base_path: Path,
    depth: int = 0,
    ignore: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Map subdirectory names to their Path objects, at a given depth.
    depth=0 means immediate children.
    """
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise ValueError(f"The path {base_path} is not a directory.")

    if ignore is None:
        ignore = ["__pycache__", ".ipynb_checkpoints"]

    def get_subdirs_at_depth(current_path: Path, current_depth: int) -> Dict[str, Path]:
        if current_depth == depth:
            return {
                subdir.name: subdir
                for subdir in current_path.iterdir()
                if subdir.is_dir()
                and subdir.name[0] not in {".", "_"}
                and subdir.name not in ignore
            }

        subdirs: Dict[str, Path] = {}
        for subdir in current_path.iterdir():
            if subdir.is_dir() and subdir.name[0] not in {".", "_"} and subdir.name not in ignore:
                subdirs.update(get_subdirs_at_depth(subdir, current_depth + 1))
        return subdirs

    return get_subdirs_at_depth(base_path, 0)


def extend_ignore_list(
    base_path: Path,
    regex_pattern: Optional[str] = None,
    use_gitignore: bool = True,
) -> List[str]:
    """
    Return a list of names/patterns to ignore, optionally:
      - all items matching regex_pattern (by name), and/or
      - patterns from .gitignore (lightweight parsing)

    NOTE: This is intentionally simple. It does NOT implement full gitignore semantics.
    """
    base_path = Path(base_path)
    ignore_list: List[str] = []

    if regex_pattern:
        regex = re.compile(regex_pattern)
        for item in base_path.rglob("*"):
            if regex.match(item.name):
                ignore_list.append(item.name)

    if use_gitignore:
        gitignore_path = base_path / ".gitignore"
        if gitignore_path.exists():
            for line in gitignore_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Normalize a little (not full semantics)
                line = line.lstrip("/").lstrip("**/").rstrip("/")
                ignore_list.append(line)

    return ignore_list


def get_directory_tree(
    base_path: Path,
    base_name: str,
    print_paths: bool = True,
    ignore: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Build a nested dict of directory structure and optionally print a tree.

    NOTE: This function prints colored output using ANSI codes if available.
    It imports colors lazily to avoid circular imports with terminal_output.
    """
    # Lazy import to avoid circular imports
    from .display import BLUE, YELLOW, RESET

    base_path = Path(base_path)
    if ignore is None:
        ignore = ["__pycache__", ".ipynb_checkpoints", "00-template.ipynb"]

    also_ignore = extend_ignore_list(base_path=base_path, regex_pattern=None, use_gitignore=True)
    ignore = list(set(ignore).union(set(also_ignore)))

    def create_path_dict(path: Path) -> Dict[str, Any]:
        path_dict: Dict[str, Any] = {}

        directories = sorted(
            [
                item
                for item in path.iterdir()
                if item.is_dir()
                and item.name not in ignore
                and not item.name.startswith(".")
            ]
        )
        files = sorted(
            [
                item
                for item in path.iterdir()
                if item.is_file()
                and item.name not in ignore
            ]
        )

        for directory in directories:
            path_dict[directory.name] = {
                "path": directory,
                "subdirectories": create_path_dict(directory),
            }

        for file in files:
            path_dict[file.name] = {"path": file}

        return path_dict

    path_structure = {base_name: {"path": base_path, "subdirectories": create_path_dict(base_path)}}

    def print_path_dict(d: Dict[str, Any], prefix: str = "", output: str = "") -> str:
        for key, value in d.items():
            if "subdirectories" in value:
                output += prefix + f"├─ {BLUE}{key}/{RESET}\n"
                output = print_path_dict(value["subdirectories"], prefix + "│  ", output)
            else:
                output += prefix + f"└─ {YELLOW}{key}{RESET}\n"
        return output

    tree_output = print_path_dict(path_structure[base_name]["subdirectories"])

    if print_paths:
        print(tree_output)

    return path_structure, tree_output


def get_variable_name(var: object) -> Optional[str]:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return None
    for name, value in frame.f_back.f_locals.items():
        if value is var:
            return name
    return None


def save_df(
    df: pd.DataFrame,
    save_as: Union[str, Path],
    tack_on: Optional[str] = None,
    index: bool = False,
) -> None:
    save_as = Path(save_as)

    if tack_on is not None:
        save_as = save_as.with_name(f"{save_as.stem}_{tack_on}{save_as.suffix}")

    ext = save_as.suffix.lower()
    if ext == ".csv":
        df.to_csv(save_as, index=index)
    elif ext == ".xlsx":
        df.to_excel(save_as, index=index)
    elif ext == ".html":
        df.to_html(save_as, index=index)
    elif ext == ".json":
        df.to_json(save_as, orient="records")
    elif ext == ".png":
        plt.axis("off")
        plt.table(cellText=df.values, colLabels=df.columns, loc="center")
        plt.savefig(save_as)
    elif ext == ".txt":
        save_as.write_text(str(df.to_dict()), encoding="utf-8")
    else:
        print(f"Unsupported file extension: {ext}. Please use csv, xlsx, html, json, png, or txt.")
