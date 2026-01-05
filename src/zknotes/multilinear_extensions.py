from typing import Optional, Union, List, Dict, Tuple, Set, Callable, Sequence, Any
from utils import count_calls, print_header, display_aligned
from utils import RED, GREEN, YELLOW, PINK, BLUE, PURPLE, RESET
import numpy as np
from itertools import product
import random
import re
from sympy import symbols, Poly, isprime, sympify, degree
from sympy.polys.domains import GF
from sympy.polys.domains import Domain
from sympy.polys.domains.modularinteger import ModularInteger
from sympy.polys.domains.finitefield import FiniteField
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sum_check import stringify

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
        X = symbols(f"x_:v")
    if Y is None:
        Y = symbols(f"y_:v")

    # Generate the arrays of polynomials for x and y variables
    x: np.ndarray[Poly] = np.array([Poly(X[j], X, domain=field) for j in range(v)])
    y: np.ndarray[Poly] = np.array([Poly(Y[j], Y, domain=field) for j in range(v)])

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
        L_0 = Poly(1 - X[j], X[j], domain=field)
        L_1 = Poly(X[j], X[j], domain=field)
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
        term = f_val
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

def multilinear_extension_example(show_result: bool = True, compare: bool = False):
    mle: List[Poly] = []
    grid: List[np.ndarray] = []
    pee = []
    vee = []
    loop_counter: int = -1
    carry_on: bool = True
    while carry_on:
        loop_counter += 1
        if loop_counter < 2:
            print(f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET}")
            p = 11
            v = 2
        else:
            p = input(f"Enter a prime number p (we will work over GF(p)): ")
            p = int(p)
            # Check if p is a prime number
            if not isprime(p):
                raise ValueError("The input 'p' must be a prime number.")

            v = input(f"Enter a positive integer v (we will extend a function defined on {{0,1}}**v: ")
            v = int(v)
            # Check if v is a positive integer
            if not isinstance(v, int) or v <= 0:
                raise ValueError("The input 'v' must be a positive integer.")

        pee.append(p)
        vee.append(v)

        # Set the field as GF(p) with symmetric=False
        field = GF(p, symmetric=False)
        print(f"\nWe work over {field}. We extend a {field}-valued function defined on {{0,1}}**{v} to one defined on {field}**{v}.\n")

        # Initialize a dictionary to store function values for each tuple in {0, 1}^v
        func_values: Dict[Tuple[int, ...], field] = {}

        if loop_counter < 2:
            # Default examples
            func_values[(0, 0)] = field(1)
            print(f"The function value for input {(0,0)} is: {1}")
            func_values[(0, 1)] = field(2)
            print(f"The function value for input {(0, 0)} is: {2}")
            func_values[(1, 0)] = field(3)
            print(f"The function value for input {(0, 0)} is: {3}")
            func_values[(1, 1)] = field(4) - field(loop_counter)
            print(f"The function value for input {(0, 0)} is: {4 - loop_counter}")
        else:
            # Iterate over all binary tuples in {0, 1}^v
            for bitstring in product([0, 1], repeat=v):
                # Request user input for the value of the function at the current tuple
                user_input = input(f"Enter the function value for input {bitstring}: ")

                # Validate that the input is an integer
                try:
                    user_value = int(user_input)
                except ValueError:
                    raise ValueError("The function value must be an integer.")

                # Convert the integer to an element of the field
                func_values[bitstring] = field(user_value)

        # Define 'explicit_function' based on the dictionary
        def explicit_function(inputs: Tuple[int, ...]) -> field:
            if inputs in func_values:
                return func_values[inputs]
            else:
                raise ValueError(f"No function value defined for input {inputs}")

        # Run multilinear_extension with the user's field, v, and func=explicit_function
        extended_func = multilinear_extension(field=field, v=v, func=explicit_function)
        mle.append(extended_func)
        eval_array = evaluation_array(polynomial=extended_func)
        grid.append(eval_array)

        if show_result:
            X = symbols(f"x_:{v}")
            X = stringify(X)
            print(f"\nThe multilinear extension of this function is:\n\n{YELLOW}f̃({X}) = {mle[-1].as_expr()}{RESET}")

        if show_result and v <= 2 and p < 50:
            print(f"\nIn the following array, entry (i,j) is f̃(i,j):\n")
            print(grid[-1])

        if loop_counter%2 == 1 and compare:
            if all(p == pee[-2] for p in pee[-2:]) and all(v == vee[-2] for v in vee[-2:]):
                p = pee[-2]
                v = vee[-2]
                # Initialize a new array with the same shape, using dtype=object to store tuples
                tuple_array = np.empty(grid[-2].shape, dtype=object)
                # Fill in the new array with tuples of corresponding elements, converting to native int
                for index, _ in np.ndenumerate(grid[-2]):
                    tuple_array[index] = (int(grid[-2][index]), int(grid[-1][index]))
                print(f"\nThe multilinear extension of the first function ({BLUE}EXAMPLE {loop_counter}{RESET}) is: {YELLOW}{mle[-2].as_expr()}{RESET}.")
                print(f"\nThe multilinear extension of the second function ({BLUE}EXAMPLE {loop_counter + 1}{RESET}) is: {YELLOW}{mle[-1].as_expr()}{RESET}.")
                print(f"\nThe multilinear extensions can agree on at most {v*p**(v-1)} out of {p**v} points.")
                if v <= 2 and p < 50:
                    print(
                        f"\nIndeed, they agree on {GREEN}{(grid[-2] == grid[-1]).sum()} points{RESET}, as shown below.")
                    print(f"\nEntry in row i, column j is a tuple where the first (second) coordinate is the evaluation of the first (second) MLE at (i,j).\n")
                    # Format and print the array
                    for i in range(tuple_array.shape[0]):
                        for j in range(tuple_array.shape[1]):
                            element = tuple_array[i, j]
                            if len(set(element)) == 1:  # All elements in the tuple are equal
                                print(f"{GREEN}{element}{RESET}", end=" ")
                            else:  # Elements are not all equal
                                print(f"{RED}{element}{RESET}", end=" ")
                        print()  # Newline after each row
                else:
                    print(
                        f"\nIndeed, they agree on {(grid[-2] == grid[-1]).sum()} points.")
            else:
                raise ValueError(f"Enter exactly two functions, with the same p and v values.")

        if loop_counter > 0:
            again = input("\nAnother example? (y/n)")
            if again == 'n':
                carry_on = False
            else:
                print(f"\n{BLUE}EXAMPLE {loop_counter + 1 + 1}.{RESET}")
    # Return the extended function as a Poly over the field
    return mle, grid


def evaluation_array(polynomial: Poly) -> np.ndarray:
    """Evaluate a polynomial over all points in GF(p)^v and store in a numpy array."""
    F = GF(polynomial.domain.mod, symmetric=False)
    v = len(polynomial.gens)
    vars = symbols(f"x_:{v}")

    elements = list(F(a) for a in range(F.mod))

    shape = (len(elements),) * v
    results = np.zeros(shape, dtype=int)

    for point in product(elements, repeat=v):
        eval_result = F(polynomial.eval(dict(zip(vars, point))))
        point = [int(x) for x in point]
        results[*point] = int(eval_result)

    return results