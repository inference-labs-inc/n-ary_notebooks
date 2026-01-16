from typing import Optional, Union, List, Dict, Tuple, Callable, Sequence, Any
from .utils import RED, GREEN, YELLOW, BLUE, RESET
from .sum_check import stringify
import numpy as np
from itertools import product
from sympy import symbols, Poly, isprime
from sympy.polys.domains import GF
from sympy.polys.domains import Domain
from sympy.polys.domains.modularinteger import ModularInteger
from sympy.polys.domains.finitefield import FiniteField
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol

from .terminal_output import TerminalOutput, out

"""
CONSTRUCTING MLE
"""
def tilde_beta_poly(
        field: FiniteField,
        v: int,
        X: Union[None, Tuple[Symbol]] = None,
        Y: Union[None, Tuple[Symbol]] = None
) -> Poly:
    """
    Computes the polynomial tilde_beta for a given finite field and number of variables.

    The polynomial tilde_beta is constructed from the terms (1 - x_j)(1 - y_j) + x_j * y_j
    for each pair of variables x_j and y_j, and then the product of these terms is taken over
    all variables to form the final polynomial.

    Args:
        field (FiniteField): The finite field over which the polynomial is defined.
                              It must be a prime-order field.
        v (int): The number of variables.
        X (Tuple[Symbol], optional): The symbolic variables for the x terms (default: None).
        Y (Tuple[Symbol], optional): The symbolic variables for the y terms (default: None).

    Returns:
        Poly: The final polynomial tilde_beta as the product of terms (1 - x_j)(1 - y_j) + x_j * y_j.

    Raises:
        ValueError: If the field's order is not prime.
    """
    # Ensure the field is prime
    if not isprime(field.mod):
        raise ValueError("Sorry, we're using SymPy and can only handle prime-order fields at present.")

    # Re-initialize field as GF(field.mod, symmetric=False)
    field = GF(field.mod, symmetric=False)

    # Default values for X and Y if not provided
    if X is None:
        X = symbols(f"x_:{v}")
    if Y is None:
        Y = symbols(f"y_:{v}")

    # Generate the arrays of polynomials for x and y variables
    XY = X + Y
    x: np.ndarray[Poly] = np.array([Poly(X[j], XY, domain=field) for j in range(v)])
    y: np.ndarray[Poly] = np.array([Poly(Y[j], XY, domain=field) for j in range(v)])

    # Compute the jth term for each pair (x_j, y_j) using (1 - x_j)(1 - y_j) + x_j * y_j
    jth_term: np.ndarray[Poly] = (1 - x) * (1 - y) + x * y

    # The final tilde_beta polynomial is the product of all jth terms
    tilde_beta: Poly = jth_term.prod()

    return tilde_beta

def multilinear_extension(
    field: FiniteField,
    v: int,
    func: Callable[[Tuple[int, ...]], int],
) -> Poly:
    """
    Creates the multilinear extension of a function on the Boolean hypercube {0,1}^v.

    Args:
        field (FiniteField): The finite field over which the function is defined.
        v (int): The number of variables.
        func (Callable[[Tuple[int, ...]], int]): A Boolean function defined on the hypercube {0,1}^v,
                                                 which is to be extended.

    Returns:
        Poly: The multilinear polynomial corresponding to the function 'func' on {0,1}^v.
    """
    # Initialize the finite field
    field = GF(field.mod, symmetric=False)

    if v == 0:
        # Degenerate case: return a constant polynomial
        b = ()
        fb = field(func(b))
        # Introduce a dummy variable x_0
        X = symbols(f"x_:{v+1}")
        return Poly(fb, X, domain=field)
    else:
        # Create symbols for the x and y variables
        X = symbols(f"x_:{v}")
        Y = symbols(f"y_:{v}")

        # Compute the tilde_beta polynomial
        tilde_beta = tilde_beta_poly(field=field, v=v, X=X, Y=Y)

        # Initialize the multilinear extension polynomial to zero
        tilde_f = Poly(0, X, domain=field)

        # Iterate over all possible Boolean values for the v variables
        B = product([0, 1], repeat=v)
        for b in B:
            # Evaluate the function at the point b
            fb = field(func(b))

            # Create a mapping from y variables to the values in b
            b_point = {Y[j]: b[j] for j in range(v)}

            # Compute the term fb * tilde_beta evaluated at b
            term = fb * tilde_beta.subs(b_point)
            tilde_f += term

        # # Convert the final expression to a polynomial
        # tilde_f = Poly(tilde_f_expr, X, domain=field)

        return tilde_f


"""
START ALTERNATE APPROACH TO CONSTRUCTING MLE (MORE OR LESS EQUIVALENT TO ABOVE)
"""
# Create Lagrange basis polynomials for two points, 0 and 1
def lagrange_basis(v: int,
                   field: FiniteField) -> List[Tuple[Poly, Poly]]:
    """Creates Lagrange basis polynomials for the points 0 and 1 in each variable."""
    field = GF(field.mod, symmetric=False)
    X = symbols(f"x_:{v}")
    basis = []
    for j in range(v):
        # Basis polynomials are (1 - x_j) for point 0 and x_j for point 1
        L_0 = Poly(1 - X[j], X, domain=field)
        L_1 = Poly(X[j], X, domain=field)
        basis.append((L_0, L_1))
    return basis


def multilinear_extension_using_lagrange_basis(
        field: FiniteField,
        v: int,
        func: Callable[[Sequence[Union[int, Integer, ModularInteger]]], Union[int, Integer, ModularInteger]],
) -> Poly:
    """Creates the multilinear extension of a function on the Boolean hypercube {0,1}^v."""

    # Set the field with non-negative representatives
    field = GF(field.mod, symmetric=False)

    # Define symbolic variables
    X = symbols(f"x_:{v}")

    basis=lagrange_basis(v=v,
                         field=field,)

    # Initialize tilde_f as the zero polynomial over the given field
    tilde_f = Poly(0, X, domain=field)

    # Iterate over all combinations in the hypercube {0,1}^v
    for binary_vector in product([0, 1], repeat=v):
        # Compute func(binary_vector) for current binary input
        f_val = field(func(binary_vector))

        # Compute the product of Lagrange basis polynomials at `point`
        term = Poly(f_val, X, domain=field)
        for i in range(v):
            term *= basis[i][binary_vector[i]]

        # Add the term to tilde_f
        tilde_f += term

    return tilde_f

"""
END ALTERNATE APPROACH TO CONSTRUCTING MLE
"""

"""
CREATE EXAMPLES
"""

def multilinear_extension_example(
    show_result: bool = True,
    compare: bool = False,
    *,
    out: TerminalOutput = out,
) -> Optional[Tuple[List[Poly], List[np.ndarray]]]:
    """
    Interactive demo: build MLEs of functions f : {0,1}^v -> GF(p).

    - EXAMPLE 1 and 2 are fixed defaults over GF(11), v=2.
    - After that, user can add more examples interactively.
    - If compare=True, compares consecutive examples when p and v match.
    """
    mle: List[Poly] = []
    grid: List[np.ndarray] = []
    pee: List[int] = []
    vee: List[int] = []

    loop_counter: int = 0

    while True:
        # ----------------------------
        # Choose (p, v) for this example
        # ----------------------------
        if loop_counter in {0, 1}:
            p = 11
            v = 2
            out.print(f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET}")
        else:
            out.print(f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET}")
            p = int(out.input("Enter a prime number p (we will work over GF(p)): ").strip())
            if not isprime(p):
                out.print(f"{RED}Invalid input: p must be prime.{RESET}")
                continue

            v = int(out.input("Enter a positive integer v (we extend f : {0,1}^v -> GF(p)): ").strip())
            if v <= 0:
                out.print(f"{RED}Invalid input: v must be a positive integer.{RESET}")
                continue

        pee.append(p)
        vee.append(v)

        field = GF(p, symmetric=False)

        # nicer formatting than **v
        out.print(f"\nWe extend a function f : {{0,1}}^{v} -> {field} to a function f̃ : {field}^{v} -> {field}.\n")

        # ----------------------------
        # Collect function values on {0,1}^v
        # ----------------------------
        func_values: Dict[Tuple[int, ...], Any] = {}

        if loop_counter == 0:
            func_values[(0, 0)] = field(1); out.print("The function value for input (0, 0) is: 1")
            func_values[(0, 1)] = field(2); out.print("The function value for input (0, 1) is: 2")
            func_values[(1, 0)] = field(3); out.print("The function value for input (1, 0) is: 3")
            func_values[(1, 1)] = field(4); out.print("The function value for input (1, 1) is: 4")

        elif loop_counter == 1:
            # Slightly different second default so the compare demo is interesting
            func_values[(0, 0)] = field(1); out.print("The function value for input (0, 0) is: 1")
            func_values[(0, 1)] = field(2); out.print("The function value for input (0, 1) is: 2")
            func_values[(1, 0)] = field(3); out.print("The function value for input (1, 0) is: 3")
            func_values[(1, 1)] = field(3); out.print("The function value for input (1, 1) is: 3")

        else:
            for bitstring in product([0, 1], repeat=v):
                user_input = out.input(f"Enter the function value for input {bitstring}: ").strip()
                try:
                    user_value = int(user_input)
                except ValueError:
                    out.print(f"{RED}Invalid input: function values must be integers.{RESET}")
                    break
                func_values[bitstring] = field(user_value)
            else:
                # only runs if the for-loop did NOT break
                pass
            # If the loop broke, restart this example without incrementing loop_counter
            if len(func_values) != 2**v:
                continue

        def explicit_function(inputs: Tuple[int, ...]):
            return func_values[inputs]

        # ----------------------------
        # Compute MLE and (optionally) show grid
        # ----------------------------
        extended_func = multilinear_extension(field=field, v=v, func=explicit_function)
        mle.append(extended_func)

        eval_array = evaluation_array(polynomial=extended_func)
        grid.append(eval_array)

        if show_result:
            X = stringify(symbols(f"x_:{v}"))
            out.print(f"\nThe multilinear extension of this function is:\n\n{YELLOW}f̃({X}) = {extended_func.as_expr()}{RESET}")

        if show_result and v <= 2 and p < 50:
            out.print(f"\nIn the following array, entry (i,j) is f̃(i,j):\n")
            out.print(eval_array)

        # ----------------------------
        # Compare consecutive examples if requested and compatible
        # ----------------------------
        if compare and len(mle) >= 2:
            if pee[-1] == pee[-2] and vee[-1] == vee[-2]:
                p_cmp = pee[-1]
                v_cmp = vee[-1]

                out.print(f"\nThe multilinear extension of the first function "
                      f"({BLUE}EXAMPLE {loop_counter}{RESET}) is: {YELLOW}{mle[-2].as_expr()}{RESET}.")
                out.print(f"\nThe multilinear extension of the second function "
                      f"({BLUE}EXAMPLE {loop_counter + 1}{RESET}) is: {YELLOW}{mle[-1].as_expr()}{RESET}.")
                out.print(f"\nThe multilinear extensions can agree on at most {v_cmp * p_cmp**(v_cmp - 1)} "
                      f"out of {p_cmp**v_cmp} points.")

                if v_cmp <= 2 and p_cmp < 50:
                    agree = (grid[-2] == grid[-1]).sum()
                    out.print(f"\nIndeed, they agree on {GREEN}{agree} points{RESET}, as shown below.\n")

                    tuple_array = np.empty(grid[-2].shape, dtype=object)
                    for index, _ in np.ndenumerate(grid[-2]):
                        tuple_array[index] = (int(grid[-2][index]), int(grid[-1][index]))

                    for i in range(tuple_array.shape[0]):
                        row_parts = []
                        for j in range(tuple_array.shape[1]):
                            a, b = tuple_array[i, j]
                            if a == b:
                                row_parts.append(f"{GREEN}{(a, b)}{RESET}")
                            else:
                                row_parts.append(f"{RED}{(a, b)}{RESET}")
                        out.print(" ".join(row_parts))
            else:
                out.print(f"\n{YELLOW}Compare skipped: last two examples used different (p, v).{RESET}")

        # ----------------------------
        # Prompt to continue (ALWAYS reachable)
        # ----------------------------
        again = out.input("\nAnother example? (y/n) ").strip().lower()
        loop_counter += 1
        if again != "y":
            break       
    return mle, grid

def evaluation_array(polynomial: Poly) -> np.ndarray:
    """
    Evaluate a polynomial over all points in GF(p)^v and store in a numpy array.

    Important: use `polynomial.gens` (not freshly-created symbols), otherwise SymPy may not
    substitute/evaluate correctly and can become extremely slow.
    """
    F = GF(int(polynomial.domain.mod), symmetric=False)
    gens = polynomial.gens
    v = len(gens)

    elements = [F(a) for a in range(F.mod)]
    shape = (len(elements),) * v
    results = np.zeros(shape, dtype=int)

    # Evaluate on every point
    for point in product(elements, repeat=v):
        # dict maps the *actual* generators to field elements
        subs = dict(zip(gens, point))
        val = polynomial.eval(subs)

        # Ensure we coerce into GF(p) then into int
        valF = F(val)
        idx = tuple(int(x) for x in point)
        results[idx] = int(valF)

    return results
