from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Set

from sympy import Poly, symbols, isprime
from sympy.core.symbol import Symbol
from sympy.polys.domains import GF
from sympy.polys.domains import Domain
from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.modularinteger import ModularInteger

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from .display import TerminalOutput, out as default_out

__all__ = [
    # parsing / gens
    "infer_gens",
    "parse_poly",
    "parse_poly_in_ring",
    "coerce_poly_to_gens",
    "align_two_polys",
    # prompting
    "choose_prime_field",
    "choose_polynomial",
    "choose_polynomial_in_ring",
    # field elements
    "input_random_field_element",
    "random_field_element",
    # formatting
    "poly_to_string",
]



# ============================================================
# Regexes + small data types
# ============================================================

# Matches:
#   X_0, X_{0}, X_12, X_{12}, X_neg1, X_neg12
# Base is a single Roman letter to avoid matching e.g. "sin_0"
_INDEXED_RE = re.compile(
    r"\b(?P<base>[A-Za-z])_(?P<idx>\d+|neg\d+)\b"
)

# single Roman letter as a standalone token (avoids matching "sin", "rho", etc.)
_SINGLE_LETTER_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z])(?![A-Za-z0-9_])")

_IDENT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\b")
_LETTER_RUN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]{2,})(?![A-Za-z0-9_])")

# handle negative subscripts
_NEG_SUB_RE = re.compile(r"(?P<base>[A-Za-z]+)_\{-(?P<idx>\d+)\}")
_POS_SUB_RE = re.compile(r"(?P<base>[A-Za-z]+)_\{(?P<idx>\d+)\}")
# also catch X_-1 without braces (best-effort)
_NEG_SUB_NOBRACE_RE = re.compile(r"(?P<base>[A-Za-z]+)_(?P<idx>-\d+)\b")
# Treat implicit multiplication by a numeric literal as explicit:
#   2x -> 2*x
#   12X_0 -> 12*X_0
_NUMBER_LETTER_RE = re.compile(r"(\d)\s*([A-Za-z])")

@dataclass(frozen=True)
class IndexedVar:
    base: str
    index: int

    @property
    def name(self) -> str:
        if self.index < 0:
            return f"{self.base}_neg{abs(self.index)}"
        return f"{self.base}_{self.index}"

# ============================================================
# String normalization (syntax sugar -> SymPy-friendly)
# ============================================================
    
def _normalize_subscripts(s: str) -> str:
    # X_{11} -> X_11 (fine)
    s = _POS_SUB_RE.sub(lambda m: f"{m.group('base')}_{m.group('idx')}", s)

    # X_{-1} -> X_neg1
    s = _NEG_SUB_RE.sub(lambda m: f"{m.group('base')}_neg{m.group('idx')}", s)

    # Best-effort: X_-1 -> X_neg1
    # (This is ambiguous in general, but if user writes it, they mean a subscript.)
    s = _NEG_SUB_NOBRACE_RE.sub(lambda m: f"{m.group('base')}_neg{m.group('idx')[1:]}", s)

    return s

def _normalize_poly_string(s: str) -> str:
    """
    Normalize common user-friendly syntax to SymPy-friendly syntax.

    - Whitespace trimmed
    - Subscripts normalized:
        X_{11} -> X_11
        X_{-2} -> X_neg2
        X_-2   -> X_neg2   (best-effort)
    NOTE: '^' is handled later by the convert_xor transform in parse_expr.
    """
    s = s.strip()
    s = _normalize_subscripts(s)
    return s

# ============================================================
# Variable extraction + generator ordering (infer_gens)
# ============================================================

def _extract_variable_names(
    poly_str: str,
    *,
    allowed_identifiers: Optional[Set[str]] = None,
) -> Tuple[List[IndexedVar], List[str], List[str]]:
    """
    Returns:
      indexed_vars, single_letter_vars, multi_letter_vars
    """
    # Normalize subscripts first (X_{-1} -> X_neg1, etc.), then general poly string cleanup
    s = _normalize_poly_string(_normalize_subscripts(poly_str))

    # Make digit-letter implicit multiplication visible to regex scanning: 2x -> 2*x
    s = _NUMBER_LETTER_RE.sub(r"\1*\2", s)

    indexed: Dict[Tuple[str, int], IndexedVar] = {}
    for m in _INDEXED_RE.finditer(s):
        base = m.group("base")
        idx_raw = m.group("idx")
        if idx_raw.startswith("neg"):
            idx = -int(idx_raw[3:])
        else:
            idx = int(idx_raw)
        indexed[(base, idx)] = IndexedVar(base=base, index=idx)

    # Remove indexed occurrences before scanning for plain single letters / letter runs
    s_wo_indexed = _INDEXED_RE.sub(" ", s)

    allowed_identifiers = allowed_identifiers or set()

    single: Set[str] = set()
    multi: Set[str] = set()

    # 1) standalone single letters
    for m in _SINGLE_LETTER_RE.finditer(s_wo_indexed):
        single.add(m.group(1))

    # 2) letter runs (xy, rho, sin, ...)
    for m in _LETTER_RUN_RE.finditer(s_wo_indexed):
        _, end = m.span(1)

        # function call guard: sin(, exp(, ...
        if end < len(s_wo_indexed) and s_wo_indexed[end] == "(":
            continue

        tok = m.group(1)

        # whitelisted multi-letter variable (rho, tau, ...)
        if tok in allowed_identifiers:
            multi.add(tok)
            continue

        # otherwise treat as product of single-letter vars: xy -> x*y
        single.update(tok)

    return list(indexed.values()), sorted(single), sorted(multi)

def _pretty_symbol_name(name: str) -> str:
    # X_neg12 -> X_{-12}
    m = re.fullmatch(r"([A-Za-z])_neg(\d+)", name)
    if m:
        return f"{m.group(1)}_{{-{m.group(2)}}}"
    return name

def infer_gens(
    poly_str: str,
    *,
    preferred_order: Optional[Sequence[str]] = None,
    allowed_identifiers: Optional[Set[str]] = None,
) -> List[Symbol]:
    """
    Decide a canonical generator ordering for Poly(...).

    Rules:
    1) If preferred_order is provided, it takes precedence (and is filtered to symbols present).
    2) Otherwise:
       - If indexed variables exist:
           order indexed first by (base letter, index)
           then plain letters alphabetical
       - Else:
           order plain letters alphabetical (fallback: X_0 if none)
    """
    indexed, single, multi = _extract_variable_names(poly_str, allowed_identifiers=allowed_identifiers)
    present_names: Set[str] = {iv.name for iv in indexed} | set(single) | set(multi)

    if preferred_order is not None:
        ordered = [name for name in preferred_order if name in present_names]
        # add any remaining present names deterministically
        remaining = sorted(present_names - set(ordered))
        names = ordered + remaining
        return list(symbols(names))

    if indexed:
        indexed_sorted = sorted(indexed, key=lambda iv: (iv.base, iv.index))
        names = [iv.name for iv in indexed_sorted] + sorted(multi) + sorted(single)
        return list(symbols(names))

    # No vars detected: default to X_0 (keeps old behavior)
    if not single and not multi:
        return list(symbols(["X_0"]))
    return list(symbols(sorted(multi) + sorted(single)))

# ============================================================
# Parsing (string -> Expr -> Poly)
# ============================================================

def parse_poly(
    poly_str: str,
    *,
    field: Domain,
    gens: Optional[Sequence[Symbol]] = None,
    allowed_identifiers: Optional[Set[str]] = None,
    preferred_order: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> Poly:
    """
    Parse a user string into a SymPy Poly over a given field.

    Supports:
      - caret exponentiation: x^2 (handled by convert_xor)
      - implicit multiplication: 2X_0, x(y+1), etc.
      - brace subscripts: X_{11}

    strict=True:
      - After parsing, reject any free symbols not in gens.
    """
    s = _normalize_poly_string(poly_str)

    if gens is None:
        gens = infer_gens(s, preferred_order=preferred_order, allowed_identifiers=allowed_identifiers)


    # Build locals dict for parse_expr: only these symbols are "allowed"
    local_dict = {str(sym): sym for sym in gens}

    transforms = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    expr = parse_expr(
        s,
        local_dict=local_dict,
        transformations=transforms,
        evaluate=True,
    )

    if strict:
        allowed = set(gens)
        extra = set(expr.free_symbols) - allowed
        if extra:
            extras = ", ".join(sorted(_pretty_symbol_name(str(x)) for x in extra))
            pretty_allowed = ", ".join(sorted(_pretty_symbol_name(str(x)) for x in allowed))
            raise ValueError(
                f"Unexpected symbol(s): {extras}. Allowed symbols for this polynomial are: {pretty_allowed}."
            )
        
    return Poly(expr, *gens, domain=field)

def poly_to_string(poly: Poly) -> str:
    """
    Deterministic string representation of a Poly that respects generator order.
    """
    gens = poly.gens
    terms = poly.terms()

    pieces = []
    for monom, coeff in terms:
        if coeff == 0:
            continue

        # build monomial
        mon_parts = []
        for g, exp in zip(gens, monom):
            if exp == 0:
                continue
            elif exp == 1:
                mon_parts.append(str(g))
            else:
                mon_parts.append(f"{g}^{exp}")

        mon = "*".join(mon_parts)

        if mon:
            pieces.append(f"{coeff}*{mon}" if coeff != 1 else mon)
        else:
            pieces.append(str(coeff))

    return " + ".join(pieces) if pieces else "0"

def parse_poly_in_ring(poly_str: str, *, field: Domain, gens: Sequence[Symbol]) -> Poly:
    # gens are fixed: the protocol’s ambient ring
    return parse_poly(
        poly_str,
        field=field,
        gens=gens,
        preferred_order=None,   # ignored when gens provided
        strict=True,            # reject unknown variables
    )

# ============================================================
# Poly coercion / alignment helpers
# ============================================================

def coerce_poly_to_gens(poly: Poly, gens: Sequence[Symbol], field: Domain) -> Poly:
    # Rebuild the Poly from its expression in the desired generator set
    return Poly(poly.as_expr(), *gens, domain=field)

def align_two_polys(a: Poly, b: Poly, *, gens: Sequence[Symbol], field: Domain) -> tuple[Poly, Poly]:
    return (
        coerce_poly_to_gens(a, gens, field),
        coerce_poly_to_gens(b, gens, field),
    )

# ============================================================
# Field selection + user prompting
# ============================================================

def choose_prime_field(
    *,
    out: TerminalOutput = default_out,
    prompt: str = "\nEnter a prime p:",
) -> Domain:
    p_str = out.input(prompt)
    try:
        p = int(p_str)
    except ValueError:
        raise ValueError("Invalid input: p must be an integer.")

    if not isprime(p):
        raise ValueError("Invalid input: p must be prime.")

    return GF(p, symmetric=False)

def choose_polynomial(
    field: Optional[Domain] = None,
    allowed_identifiers: Optional[Set[str]] = None,
    custom_message: Optional[str] = None,
    preferred_order: Optional[Sequence[str]] = None,
    echo_back: bool = True,
    secret_mode_for_poly: bool = False,
    *,
    out: TerminalOutput = default_out,
) -> Optional[Poly]:
    """
    Prompt for a field (if needed) + prompt for a polynomial string, then parse into Poly.

    - Accepts flexible syntax: x^2, 2X_0, X_{11}, etc.
    - Variable ordering:
        * If preferred_order is provided, it has precedence.
        * Else: indexed vars (A_i) first by (A, i), then plain letters.
    - echo_back prints a normalized representation (unless you disable it, e.g. secret_mode).
    """
    if field is None:
        field = choose_prime_field(out=out)

    # Normalize to a prime field SymPy expects (your current restriction)
    if getattr(field, "mod", None) is None:
        raise ValueError("Provided domain does not look like a prime field with attribute .mod.")
    if not isprime(int(field.mod)):
        raise ValueError("Sorry, we're using SymPy and can only handle prime fields at present.")
    field = GF(int(field.mod), symmetric=False)

    prompt = custom_message or (
        f"\nEnter your polynomial over {field} "
        f"(examples: 2*X_0^2 + X_0 X_1, x*y + z, X_{11} + 3):"
    )

    poly_str_raw = out.input(prompt, hidden=secret_mode_for_poly)
    poly_str = _normalize_subscripts(poly_str_raw)

    try:
        poly = parse_poly(
            poly_str,
            field=field,
            gens=None,
            allowed_identifiers=allowed_identifiers,
            preferred_order=preferred_order,
            strict=True,
        )
    except Exception as e:
        out.print(f"\nCould not parse polynomial: {e}")
        return None

    if echo_back and not secret_mode_for_poly:
        gens_str = ", ".join(str(g) for g in poly.gens)
        out.print(f"\nInterpreted as a polynomial in: ({gens_str})")
        out.print(f"Normalized polynomial: {poly_to_string(poly)}")

    return poly

def choose_polynomial_with_retries(
    *,
    field,
    custom_message: str,
    preferred_order=None,
    secret_mode_for_poly: bool = False,
    echo_back: bool = False,
    max_attempts: int = 3,
    out: TerminalOutput = default_out,
):
    """
    Wrapper around utils.polynomials.choose_polynomial that:
      - retries on parse failure
      - allows 'cancel' to cancel
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1

        # Let user cancel explicitly without raising
        prompt = custom_message
        if "cancel" not in prompt.lower():
            prompt = prompt.rstrip()

        s = out.input(prompt, hidden=secret_mode_for_poly).strip()
        if s.lower() == "cancel":
            return None

        # Call parse directly so we can reuse your strong error messages
        try:
            # Important: we must force gens for univariate rounds (see below)
            from gkr.utils.polynomials import parse_poly
            poly = parse_poly(
                s,
                field=field,
                gens=preferred_order,   # (see note)
                preferred_order=None,
                strict=True,
            )
            if echo_back and not secret_mode_for_poly:
                gens_str = ", ".join(str(g) for g in poly.gens)
                out.print(f"\nInterpreted as a polynomial in: ({gens_str})")
                out.print(f"Normalized polynomial: {poly.as_expr()}")
            return poly
        except Exception as e:
            if attempt < max_attempts:
                out.print(f"\nCould not parse polynomial: {e}")
                out.print(f"Try again ({attempt}/{max_attempts}).")
            else:
                out.print(f"\nCould not parse polynomial: {e}")
                out.print("No more attempts.")
                return None

# ============================================================
# Convenience: parse/choose inside a fixed ambient ring
# ============================================================

def choose_polynomial_in_ring(
    *,
    field: Domain,
    gens: Sequence[Symbol],
    custom_message: Optional[str] = None,
    echo_back: bool = True,
    secret_mode_for_poly: bool = False,
    out: TerminalOutput = default_out,
) -> Optional[Poly]:
    prompt = custom_message or f"\nEnter a polynomial over {field}:"
    s_raw = out.input(prompt, hidden=secret_mode_for_poly)
    s = _normalize_subscripts(s_raw)

    try:
        poly = parse_poly_in_ring(s, field=field, gens=gens)
    except Exception as e:
        out.print(f"\nCould not parse polynomial: {e}")
        return None

    if echo_back and not secret_mode_for_poly:
        out.print(f"\nInterpreted in the ring: F[{', '.join(map(str, gens))}]")
        out.print(f"Normalized polynomial: {poly.as_expr()}")

    return poly

# ============================================================
# Field element input
# ============================================================

def input_random_field_element(
    custom_message: Union[None, str] = None,
    max_attempts: Optional[int] = None,
    *,
    out: TerminalOutput = default_out,
) -> Optional[int]:
    """
    Backwards-compatible input routine.

    Returns:
      - int on success
      - None on cancel or too many failed attempts
    """
    attempts = 0 if max_attempts is not None else None
    prompt = (
        custom_message
        if custom_message is not None
        else "\nEnter c to cancel or select element uniformly at random from field, independent of any previous selection:"
    )

    while attempts is None or (attempts != max_attempts):
        response = out.input(prompt).strip()
        if response.lower() == "c":
            return None
        try:
            return int(response)
        except ValueError:
            if attempts is not None:
                attempts += 1
                if attempts == max_attempts:
                    out.print("\nInvalid input. No more attempts.")
                    return None
                elif attempts == max_attempts - 1:
                    prompt = "\nInvalid input. Final attempt: enter an integer, or c to cancel:"
                else:
                    prompt = "\nInvalid input. Enter an integer, or c to cancel:"
            else:
                prompt = "\nInvalid input. Enter an integer, or c to cancel:"


def _coerce_to_field_element(
    field: FiniteField,
    value: Union[int, ModularInteger],
) -> ModularInteger:
    """
    Normalize either int or ModularInteger into an element of `field`.
    """
    # SymPy ModularInteger has .val and .mod
    if isinstance(value, ModularInteger):
        # If it already matches modulus, keep its integer representative
        v = int(value)
        return field(v)

    # int
    return field(int(value))


def random_field_element(
    field: FiniteField,
    user_input: bool = False,
    custom_message: Union[None, str] = None,
    max_attempts: Optional[int] = None,
    *,
    out: TerminalOutput = default_out,
) -> Optional[ModularInteger]:
    """
    Backwards-compatible field element picker.

    - If user_input=False: returns a random element of GF(p)
    - If user_input=True: prompts user; allows cancel ('c') -> None
    """
    if user_input:
        user_response = input_random_field_element(
            custom_message=custom_message,
            max_attempts=max_attempts,
            out=out,
        )
        if user_response is None:
            return None
        return _coerce_to_field_element(field, user_response)

    # Sample uniformly in {0, ..., p-1}
    # field.mod is the prime p in SymPy FiniteField
    return field(random.randrange(0, int(field.mod)))