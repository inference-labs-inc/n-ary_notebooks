from __future__ import annotations

import os
import sys
from dataclasses import dataclass

def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False

def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _env_mode() -> str:
    # "auto" | "always" | "never"
    return os.getenv("ZKNOTES_COLOR", "auto").strip().lower()

def _windows_supports_ansi_natively() -> bool:
    # Windows Terminal / VSCode terminal typically set WT_SESSION or TERM_PROGRAM
    return (
        os.getenv("WT_SESSION") is not None
        or os.getenv("TERM_PROGRAM") is not None
        or os.getenv("ANSICON") is not None
        or os.getenv("ConEmuANSI") == "ON"
    )
def _try_enable_colorama() -> bool:
    if os.name != "nt":
        return True
    if _windows_supports_ansi_natively():
        return True
    try:
        import colorama  # type: ignore
        colorama.just_fix_windows_console()
        return True
    except Exception:
        return False

def supports_color() -> bool:
    mode = _env_mode()
    if mode == "never":
        return False
    if mode == "always":
        return _try_enable_colorama()

    # auto
    if _in_jupyter():
        return True
    if not _is_tty():
        return False
    if os.getenv("NO_COLOR") is not None:
        return False

    return _try_enable_colorama()

@dataclass(frozen=True)
class Ansi:
    RED: str = "\033[31m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    PINK: str = "\033[35m"
    BLUE: str = "\033[34m"
    PURPLE: str = "\033[35m"
    RESET: str = "\033[0m"

@dataclass(frozen=True)
class NoAnsi:
    RED: str = ""
    GREEN: str = ""
    YELLOW: str = ""
    PINK: str = ""
    BLUE: str = ""
    PURPLE: str = ""
    RESET: str = ""

def get_colors():
    return Ansi() if supports_color() else NoAnsi()

