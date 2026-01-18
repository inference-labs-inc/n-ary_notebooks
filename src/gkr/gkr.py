import numpy as np
from sympy import symbols, Poly
from sympy.polys.domains import GF
from sympy.polys.domains.modularinteger import ModularInteger
from typing import Union, Tuple


def line_vectorized(
        b: Union[np.ndarray, Tuple[Union[int, ModularInteger], ...]],
        c: Union[np.ndarray, Tuple[Union[int, ModularInteger], ...]],
        field_mod: int
) -> np.ndarray:
    """
    Constructs a parametric line passing through two points, with output polynomials over the field GF(field_mod.mod, symmetric=True).

    Parameters:
    - b (Union[np.ndarray, Tuple[Union[int, ModularInteger], ...]]):
        Coordinates of the first point, as an array or tuple of integers/ModularIntegers.
    - c (Union[np.ndarray, Tuple[Union[int, ModularInteger], ...]]):
        Coordinates of the second point, as an array or tuple of integers/ModularIntegers.
    - field_mod (ModularInteger): A ModularInteger defining the modulus of the finite field.

    Returns:
    - np.ndarray: Array of parametric equations (as `Poly` objects) for the line over the specified field.

    Raises:
    - AssertionError: If `b` and `c` have mismatched dimensions.
    """
    # Convert inputs to numpy arrays if they are tuples and cast ModularIntegers to integers
    if isinstance(b, tuple):
        b = np.array([int(elem) if isinstance(elem, ModularInteger) else elem for elem in b])
    if isinstance(c, tuple):
        c = np.array([int(elem) if isinstance(elem, ModularInteger) else elem for elem in c])

    # Ensure `b` and `c` have the same shape
    assert b.shape == c.shape, "Points `b` and `c` must have the same dimensions."

    # Validate the field modulus
    field = GF(field_mod, symmetric=False)

    # Symbolic parameter for the parametric line
    t = symbols('t')

    # Perform element-wise operations to compute the parametric line
    line_through_b_and_c = np.array([
        Poly((1 - t) * b[i] + t * c[i], domain=field) for i in range(len(b))
    ])

    return line_through_b_and_c

def compose_polynomial(
        W: Poly,
        line: np.ndarray,
) -> Poly:
    """
    Composes the polynomial W with an array of univariate polynomials `line`.

    Parameters:
    - W (Poly): A multivariate polynomial.
    - line (np.ndarray): An array of univariate polynomials to substitute.

    Returns:
    - Poly: The composed polynomial.

    Raises:
    - AssertionError: If the number of symbols in W and line are mismatched.
    """
    # Extract the variables (symbols) of W and the variable of line[0]
    x = W.gens  # Symbols from W
    t = line[0].gens[0]  # Symbol from the first polynomial in line
    assert len(x) == len(line), (
        f"Invalid input: `W` has {len(x)} indeterminates but `line` is {len(line)}-dimensional."
    )

    # Convert W to an expression
    p_composed_expr = W.as_expr()

    # Substitute each variable in W with the corresponding polynomial in line
    for xi, line_i in zip(x, line):
        p_composed_expr = p_composed_expr.subs(xi, line_i.as_expr())

    # Convert the composed expression back to a polynomial
    p_composed_poly = Poly(p_composed_expr, t, domain=W.domain)

    return p_composed_poly