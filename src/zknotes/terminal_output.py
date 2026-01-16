from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
import os
from contextlib import contextmanager

RESET = "\033[0m"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


@dataclass
class TerminalOutput:
    @contextmanager
    def atomic(self):
        old = self.enable_paging
        self.enable_paging = False
        try:
            yield
        finally:
            self.enable_paging = old

    enable_paging: bool = True
    margin_lines: int = 2
    prompt: str = ":"          # could also use "-- More --" or similar
    reset_on_prompt: bool = True

    _line_count: int = 0

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
        # How many "printable" lines we allow before paging
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
        """
        Conservative line counting:
        - count newline characters
        - if string does not end with newline, it still occupies 1 visible line
          (unless it's empty)
        """
        if s == "":
            return 0
        n_newlines = s.count("\n")
        if n_newlines == 0:
            return 1
        # If it ends with '\n', the newline count already represents lines advanced.
        # If it does not, add 1 for the final unterminated line.
        return n_newlines if s.endswith("\n") else (n_newlines + 1)

    def print(self, *args, sep: str = " ", end: str = "\n") -> None:
        s = sep.join(str(a) for a in args) + end

        # PAGE BEFORE printing, if needed
        needed_lines = self._count_lines(s)
        if self._needs_page_for(needed_lines):
            self._pause()

        # Print immediately (no buffering)
        sys.stdout.write(s)
        sys.stdout.flush()

        # Update line counter
        self._line_count += needed_lines

        # (Optional) If we *exactly* hit bottom, page now so next output starts fresh.
        # This prevents the next thing printed from beginning at the last line.
        if self._paging_enabled_now() and self._line_count >= self._available_lines():
            self._pause()

    def input(self, prompt: str = "") -> str:
        # Ensure prompt is visible and not stuck behind a pager prompt
        if self.reset_on_prompt:
            sys.stdout.write(RESET)
            sys.stdout.flush()
        return input(prompt)

    def new_section(self, min_lines: int = 1) -> None:
        """
        Call this before big section headers so they don't get cut off at bottom.
        min_lines = how many lines you want to guarantee are available next.
        """
        if min_lines < 1:
            min_lines = 1
        if self._needs_page_for(min_lines):
            self._pause()

    def flush(self) -> None:
        """
        Flush stdout and reset any internal paging counters.
        Safe to call at the end of a protocol.
        """
        try:
            sys.stdout.flush()
        except Exception:
            pass
        self._line_count = 0


out = TerminalOutput(enable_paging=True)
