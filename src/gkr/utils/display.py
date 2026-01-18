from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import getpass
import os
import shutil
import sys

from ._ansi import get_colors

# ----------------------------
# ANSI colors
# ----------------------------

_colors = get_colors()
RED = _colors.RED
GREEN = _colors.GREEN
YELLOW = _colors.YELLOW
PINK = _colors.PINK
BLUE = _colors.BLUE
PURPLE = _colors.PURPLE
RESET = _colors.RESET


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# ----------------------------
# Optional IPython helpers
# ----------------------------

try:
    from IPython.display import display as ipy_display  # type: ignore
except Exception:
    ipy_display = None


def display(*args: Any) -> None:
    if ipy_display is not None:
        for arg in args:
            if arg is not None:
                ipy_display(arg)
    else:
        for arg in args:
            if arg is not None:
                print(arg)


try:
    from IPython.display import clear_output as ipy_clear_output  # type: ignore
except Exception:
    ipy_clear_output = None


def clear_output() -> None:
    if ipy_clear_output is not None:
        ipy_clear_output(wait=True)
        return

    # Fallback: clear terminal
    os.system("cls" if os.name == "nt" else "clear")


# widgets/HTML are optional; keep stable stubs
widgets = None
HTML: Callable[..., Any]


def _stub_html(*args: Any, **kwargs: Any) -> None:
    print("HTML display is not available.")


HTML = _stub_html


def set_widgets(enable: bool = True, disable: bool = False) -> None:
    global widgets, HTML

    if disable:
        widgets = None
        HTML = _stub_html
        return

    if not enable:
        return

    try:
        from IPython import get_ipython  # type: ignore

        if get_ipython() is None:
            widgets = None
            HTML = _stub_html
            return

        from IPython.display import HTML as _HTML  # type: ignore
        import ipywidgets as _widgets  # type: ignore

        widgets = _widgets
        HTML = _HTML
    except Exception:
        widgets = None
        HTML = _stub_html


# ----------------------------
# Pretty-print helpers
# ----------------------------

def display_aligned(*strings: str, out: Optional["TerminalOutput"] = None) -> None:
    """
    Print strings containing '=' aligned on the '=' sign.
    If no string contains '=', prints all strings as-is.

    If `out` is provided, uses `out.print(...)` instead of built-in print.
    """
    printer = out.print if out is not None else print

    parts = [s.split("=", 1) for s in strings if "=" in s]
    if not parts:
        for s in strings:
            printer(s)
        return

    max_left_length = max(len(part[0].strip()) for part in parts)
    for left, right in parts:
        printer(f"{left.strip():>{max_left_length}} = {right.strip()}")


def print_header(title: str, level: int = 1, *, out: Optional["TerminalOutput"] = None) -> None:
    """
    Print a section header, using TerminalOutput paging if provided.

    level=1: big header with "=" bars above and below
    level=2: smaller header (no bars)
    """
    printer = out.print if out is not None else print

    # Conservative line budgeting so we don't split headers across pages.
    min_lines = 5 if level == 1 else 2

    if out is not None:
        out.new_section(min_lines=min_lines)
        with out.atomic():
            print_header_block(title, level=level, printer=printer)
    else:
        print_header_block(title, level=level, printer=printer)


def print_header_block(
    title: str,
    *,
    level: int = 1,
    printer: Callable[[str], None] = print,
) -> None:
    t = title.strip()
    if level == 1:
        line = "=" * len(t)
        printer(f"\n{PINK}{line}{RESET}")
        printer(f"{PINK}{t.upper()}{RESET}")
        printer(f"{PINK}{line}{RESET}")
    else:
        printer(f"\n{PINK}{t.upper()}{RESET}")


# ----------------------------
# TerminalOutput
# ----------------------------

@dataclass
class TerminalOutput:
    enable_paging: bool = True
    margin_lines: int = 2
    prompt: str = ":"          # could also use "-- More --"
    reset_on_prompt: bool = True

    _line_count: int = 0

    @contextmanager
    def atomic(self):
        old = self.enable_paging
        self.enable_paging = False
        try:
            yield
        finally:
            self.enable_paging = old

    def _term_height(self) -> int:
        return shutil.get_terminal_size(fallback=(80, 24)).lines

    def _paging_enabled_now(self) -> bool:
        if not self.enable_paging:
            return False
        if _in_notebook():
            return False
        if not sys.stdout.isatty() or not sys.stdin.isatty():
            return False
        return True

    def _available_lines(self) -> int:
        height = self._term_height()
        return max(1, height - self.margin_lines)

    def _needs_page_for(self, additional_lines: int) -> bool:
        if not self._paging_enabled_now():
            return False
        return (self._line_count + additional_lines) >= self._available_lines()

    def _pause(self) -> None:
        # Ensure we don't inherit a previous color
        if self.reset_on_prompt:
            sys.stdout.write(RESET)

        sys.stdout.write(self.prompt)
        sys.stdout.flush()

        # Read a single keypress (Windows + Unix-ish, best effort)
        try:
            if os.name == "nt":
                import msvcrt  # type: ignore
                msvcrt.getch()
            else:
                import termios, tty  # type: ignore
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            # Fallback: require Enter
            try:
                sys.stdin.readline()
            except Exception:
                pass

        # Erase the prompt (":"), keep cursor clean
        sys.stdout.write("\r \r")
        sys.stdout.flush()
        self._line_count = 0

    def _count_lines(self, s: str) -> int:
        if s == "":
            return 0
        n_newlines = s.count("\n")
        if n_newlines == 0:
            return 1
        return n_newlines if s.endswith("\n") else (n_newlines + 1)

    def print(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        s = sep.join(str(a) for a in args) + end

        needed_lines = self._count_lines(s)
        if self._needs_page_for(needed_lines):
            self._pause()

        sys.stdout.write(s)
        sys.stdout.flush()

        self._line_count += needed_lines

        if self._paging_enabled_now() and self._line_count >= self._available_lines():
            self._pause()

    def input(self, prompt: str = "", *, hidden: bool = False) -> str:
        # Ensure prompt is visible and not stuck behind a pager prompt
        if self.reset_on_prompt:
            sys.stdout.write(RESET)
            sys.stdout.flush()

        if hidden:
            return getpass.getpass(prompt)
        return input(prompt)

    def new_section(self, min_lines: int = 1) -> None:
        if min_lines < 1:
            min_lines = 1
        if self._needs_page_for(min_lines):
            self._pause()

    def flush(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        self._line_count = 0


out = TerminalOutput(enable_paging=True)
