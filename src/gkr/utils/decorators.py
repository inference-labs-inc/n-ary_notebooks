from __future__ import annotations

from typing import Callable, TypeVar, Any, cast

F = TypeVar("F", bound=Callable[..., Any])

def count_calls(func: F) -> F:
    """
    Decorator that adds a mutable attribute `call_count` to the wrapped function.

    Usage:
        @count_calls
        def f(...): ...

        f.call_count = 0   # reset whenever you want
    """
    def wrapper(*args: Any, **kwargs: Any):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0  # type: ignore[attr-defined]
    return cast(F, wrapper)