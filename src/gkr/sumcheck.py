from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple, Set, Any

from .utils import (
    count_calls,
    print_header,
    display_aligned,
    RED, GREEN, YELLOW, PINK, BLUE, PURPLE, RESET,
    TerminalOutput,
    out,
)

from itertools import product
import random
import re
import time
from sympy import symbols, Poly, isprime, sympify, degree
from sympy.polys.domains import GF
from sympy.polys.domains import Domain
from sympy.polys.domains.modularinteger import ModularInteger
from sympy.polys.domains.finitefield import FiniteField
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
import hashlib


def choose_polynomial(
    field: Optional[Domain] = None,
    custom_message: Union[None, str] = None,
    variable_names: Union[None, list] = None,
    prompt_for_k: bool = True,
    secret_mode_for_poly: bool = False,
    *,
    out: TerminalOutput = out,  
) -> Union[None, Poly]:
    """
    Prompts the user to define a finite field and input a polynomial over that field.

    If a finite field `field` is not provided, the function prompts the user to define one by inputting
    a prime number `p` and (optionally) an exponent `k`, creating the field GF(p^k).

    NEW:
      - If prompt_for_k=False, the function will NOT ask for k and will set k := 1.
        (This is useful for demos where we restrict to prime fields.)

    Args:
        field (Optional[Domain], optional): An optional SymPy field domain (e.g., GF(p)).
        custom_message (Union[None, str], optional): A custom message to display when prompting the user
                                                     for the polynomial. Defaults to None.
        variable_names (Union[None, list], optional): An optional list of variable names to include.
        prompt_for_k (bool, optional): If True, prompt for k; if False, set k=1 silently.

    Returns:
        Union[None, Poly]: A SymPy polynomial object over the field if inputs are valid; otherwise, None.

    Raises:
        ValueError: If the input prime or exponent cannot be converted to an integer,
                    or if the provided field is not a prime field.
    """
    if field is None:
        p = out.input("\nEnter a prime p:")
        try:
            p = int(p)
        except ValueError:
            raise ValueError("Invalid input: p must be an integer.")

        if not isprime(p):
            out.print("Invalid input: p must be prime.")
            return None

        if prompt_for_k:
            k = out.input("Enter an exponent k (the field will have order p**k):")
            try:
                k = int(k)
            except ValueError:
                raise ValueError("Invalid input: k must be an integer.")
        else:
            k = 1

        if k > 1:
            raise ValueError("Sorry, we're using SymPy and can only handle prime fields at present (k must be 1).")

        field = GF(p, symmetric=False)

    else:
        if getattr(field, "mod", None) is None:
            raise ValueError("Provided domain does not look like a prime field with attribute .mod.")
        if not isprime(field.mod):
            raise ValueError("Sorry, we're using SymPy and can only handle prime fields at present.")
        field = GF(field.mod, symmetric=False)

    if custom_message:
        prompt = custom_message
    else:
        prompt = (
            f"\nEnter your polynomial over {field} "
            f"(e.g. 2*X_0**2 + X_0*X_1*X_2 + X_1*X_4**3 + X_1 + X_3):"
        )

    poly_str = out.input(prompt, hidden=secret_mode_for_poly)

    detected_variables = set(re.findall(r"X_\d+", poly_str))

    if variable_names is not None:
        variable_names = sorted(set(variable_names).union(detected_variables))
    else:
        variable_names = sorted(detected_variables) if detected_variables else ["X_0"]

    variables = symbols(variable_names)
    variable_map = {name: var for name, var in zip(variable_names, variables)}

    poly_expr = sympify(poly_str, locals=variable_map)

    return Poly(poly_expr, variables, domain=field)

def is_multilinear(poly: Poly) -> bool:
    # Check if the degree of each variable in every term is at most 1
    for var in poly.gens:  # Iterate through each variable in the polynomial
        if poly.degree(var) > 1:
            return False
    return True

def total_degree_example(out=out) -> None:
    loop_counter: int = 0
    carry_on: bool = True
    while carry_on:
        if loop_counter == 0:
            # Default example
            X = symbols(f"X_:{3}")
            g = Poly((X[0] - X[1]) * (X[1] - X[2]), X, domain=GF(5, symmetric=False))
        else:
            # Do NOT prompt for k in this demo (prime fields only)
            g = choose_polynomial(prompt_for_k=False)

        multilinear: str = 'multilinear' if is_multilinear(g) else 'non-multilinear'
        X_tuple = ', '.join([str(X) for X in g.gens])
        out.print(
            f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET} The polynomial \n\n"
            f"g({X_tuple}) = {g.as_expr()} \n\n"
            f"is a {multilinear} polynomial in {g.domain}[{X_tuple}]."
        )
        out.print(f"\nThe total degree of g is deg(g) = {g.total_degree()}")
        for X in g.gens:
            out.print(f"\nThe degree of {X} in g is deg_{X}(g) = {degree(g, X)}.")
        again = out.input("\nAnother example? (y/n)")
        loop_counter += 1
        if again == 'n':
            carry_on = False
        else:
            out.print("")


def roots(poly: Poly, time_out: Optional[float] = 60) -> Tuple[Set[Tuple[int, ...]], bool]:
    """
    Finds the roots of a polynomial over its defined finite field, with an optional timeout.

    This function iterates over all possible v-tuples of elements in the field
    associated with the polynomial's domain and evaluates the polynomial at each
    tuple. If the polynomial evaluates to zero at a tuple, that tuple is added
    to the set of roots. If the specified timeout is exceeded, the search is
    terminated.

    Args:
        poly (Poly): The polynomial to evaluate, represented as a SymPy Poly object
                     defined over a finite field.
        time_out (float, optional): Maximum time allowed for root finding in seconds.
                                    Defaults to None (no timeout).

    Returns:
        Tuple[Set[Tuple[int, ...]], bool]: A tuple containing the set of roots and a boolean
                                           indicating if the loop was terminated early.
    """
    # Extract the finite field from the polynomial's domain
    field = poly.domain
    if field.has_CharacteristicZero:
        raise ValueError("The polynomial must be defined over a finite field.")
    if not isprime(field.mod):
        raise ValueError("Sorry, 'we're using SymPy and can only handle prime fields at present.")

    # Get the list of variables in the polynomial
    variables = poly.gens
    num_vars = len(variables)

    # Prepare the set to store roots
    root_set = set()

    # Initialize timing if time_out is specified
    if time_out is not None:
        tic = time.time()

    # Iterate over all possible v-tuples of elements in the field
    for values in product(range(field.mod), repeat=num_vars):
        # Check if the time limit has been exceeded
        if time_out is not None and time.time() - tic > time_out:
            return root_set, True  # Early termination

        # Create a dictionary mapping variables to values in the tuple
        eval_point = {var: field(val) for var, val in zip(variables, values)}

        # Evaluate the polynomial at the current tuple
        if poly.eval(eval_point) == 0:
            root_set.add(values)

    return root_set, False  # Completed without early termination

def _domain_power_str(g: Poly, v: int) -> str:
    """
    Human-friendly string for the ambient space of tuples.

    We want to convey: tuples in (GF(p))^v (Cartesian power),
    and avoid the misleading GF(p)**(v) which reads like GF(p^v).
    """
    return f"({g.domain})^({v})"


def _fmt_prob_fraction(num_roots: int, p: int, v: int) -> str:
    """
    Format the probability as an exact fraction plus a decimal approximation.
    """
    denom = p**v
    prob = num_roots / denom
    return f"{num_roots}/{p}**{v} = {prob}"


def _fmt_sz_bound(d: int, p: int) -> str:
    """
    Format the Schwartz-Zippel upper bound d/#F as exact fraction plus decimal.
    """
    bound = d / p
    return f"{d}/{p} = {bound}"


def roots_example(time_out: Optional[float] = None, out=out) -> None:
    """
    Interactive demo of Schwartz-Zippel: count roots (exactly if feasible, otherwise partially).

    - Prints the ambient space as (GF(p))^(v) to avoid confusion with extension fields.
    - When enumeration completes, compares observed probability to the SZ bound (with if/then).
    - When timed out, clearly labels the probability as a lower bound and reminds SZ is an upper bound.
    """
    loop_counter: int = 0
    carry_on: bool = True

    while carry_on:
        # --- Choose polynomial ---
        if loop_counter == 0:
            X = symbols("X_:3")
            g = Poly(X[0] * X[1] + X[2] ** 2, X, domain=GF(41, symmetric=False))
        else:
            g = choose_polynomial(prompt_for_k=False)
            if g is None:
                # User entered invalid input; restart loop without incrementing counter.
                continue

        zeros, ran_out_of_time = roots(poly=g, time_out=time_out)
        zeros = list(zeros)

        v = len(g.gens)
        p = int(g.domain.mod)
        d = int(g.total_degree())

        space_str = _domain_power_str(g, v)
        r_tuple = ", ".join([f"r_{j}" for j in range(v)])

        lhs_str = _fmt_prob_fraction(len(zeros), p, v)
        rhs_str = _fmt_sz_bound(d, p)

        # --- Header + probability statement ---
        if ran_out_of_time:
            out.print(
                f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET} "
                f"In {GREEN}{time_out} seconds{RESET}, we found {len(zeros)} roots of {g.as_expr()} in {space_str}."
            )
            out.print(f"\n{GREEN}There may be more roots, so what follows is not a verification of Schwartz-Zippel.{RESET}")
            out.print(
                f"\nThus, if each of {r_tuple} is chosen independently and uniformly at random from {g.domain}, "
                f"then g({r_tuple}) = 0 with probability {GREEN}at least{RESET} {lhs_str} "
                f"(this is a lower bound). Schwartz-Zippel gives an upper bound."
            )
        else:
            out.print(
                f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET} "
                f"The polynomial {g.as_expr()} has {len(zeros)} roots in {space_str}."
            )
            out.print(
                f"\nThus, if each of {r_tuple} is chosen independently and uniformly at random from {g.domain}, "
                f"then g({r_tuple}) = 0 with probability {lhs_str}."
            )

        # --- Degree + SZ bound ---
        out.print(f"\nThe polynomial has total degree d = {d}, and the Schwartz-Zippel bound is {rhs_str}.")

        # If we enumerated all roots, compare observed probability to the bound.
        if not ran_out_of_time:
            prob = len(zeros) / (p**v)
            bound = d / p
            eps = 1e-12
            if prob < bound - eps:
                rel = "below"
            elif prob > bound + eps:
                rel = "above"
            else:
                rel = "equal to (within rounding)"
            out.print(f"\nObserved probability is {rel} the Schwartz-Zippel upper bound.")

        # --- Print some roots ---
        truncate: int = 10
        if len(zeros) > truncate:
            out.print(f"\nHere are {truncate} of the roots:\n")
            to_print = [f"g{z} = 0" for z in zeros[:truncate]]
        else:
            if ran_out_of_time:
                out.print(f"\nHere are the roots we found before running out of time:\n")
            else:
                out.print(f"\nHere are the roots:\n")
            to_print = [f"g{z} = 0" for z in zeros]

        display_aligned(*to_print, out=out)

        # --- Continue? ---
        again = out.input("\nAnother example? (y/n) ").strip().lower()
        loop_counter += 1
        if again == "n":
            carry_on = False

"""
SUM CHECK PROTOCOL
"""

def sum_check_example(out=out) -> None:
    loop_counter: int = 0
    carry_on: bool = True
    while carry_on:
        if hasattr(out, "flush"):
            out.flush()
        out.print(f"\n{BLUE}EXAMPLE {loop_counter + 1}.{RESET} ")
        if loop_counter == 0:
            # Default example
            v: int = 3
            X: Symbol = symbols(f"X_:{v}")
            F: FiniteField = GF(5, symmetric=False)
            multivariate_init: Poly = Poly((X[0] - X[1]) * (X[1] - X[2]), X, domain=F)
            # Calculate the true sum by evaluating the polynomial over all Boolean inputs
            H = F(sum(
                multivariate_init.eval({X[k]: b_k for k, b_k in enumerate(b)})
                for b in product([F(0), F(1)], repeat=v)
            ))

            # Call the recursive sum-check protocol
            sum_check_recursion(
                g_init=multivariate_init,
                g=multivariate_init,
                H_star=H,
            )
        else:
            sum_check()
        again = out.input("\nAnother example? (y/n)").strip().lower()
        loop_counter += 1
        if again == 'n':
            carry_on = False

def sum_check(out=out, 
              *, 
              use_commitment: bool = False,
              secret_mode: Optional[bool] = None,) -> Optional[bool]:
    """
    Sets up and initiates the sum-check protocol using user input and interactive choices.

    This function:
    - Prompts the user to choose a polynomial over a prime field GF(p).
    - Verifies that the chosen field is prime.
    - Asks the user if the protocol should run interactively and if they wish to act as verifier or prover.
    - If not acting as a prover, determines a scenario in which the prover may be dishonest.
    - Computes the correct sum of the polynomial over the Boolean hypercube.
    - If the user is the prover, it allows them to provide a claimed sum, potentially introducing dishonest behavior.
    - Finally, it calls `sum_check_recursion` to run the recursive steps of the sum-check protocol.

    Returns:
        Optional[bool]:
            - True if the sum-check protocol completes successfully (verifier accepts).
            - False if the verification fails (verifier rejects).
            - None if inputs are invalid or conditions for running are not met.
    """
    # Reset call count at the start of a new protocol run
    sum_check_recursion.call_count = 0
    
    # Start fresh so the first header doesn't appear mid-page
    if hasattr(out, "flush"):
        out.flush()
    
    # Prompt user for interactive mode
    interactive_input = out.input("\nDo you want this to be interactive? (y/n): ")
    user_input = interactive_input.strip().lower() == 'y'
    if not user_input:
        out.print("\nAlthough you've selected non-interative mode, responses to a few prompts are still required...")

    # Initialize user_is_verifier and user_is_prover booleans as False (will update according to user responses below)
    user_is_verifier: bool = False
    user_is_prover: bool = False

    if user_input:
        user_selection_prover = out.input("Do you want to act as prover? (y/n): ")
        user_selection_verifier = out.input("Do you want to act as verifier? (y/n): ")
        user_is_prover = user_selection_prover.strip().lower() == 'y'
        user_is_verifier = user_selection_verifier.strip().lower() == 'y'

    if secret_mode is None:
        secret_mode = (user_is_verifier and not user_is_prover)

    # Get the polynomial from the user
    multivariate_init = choose_polynomial(prompt_for_k=False, 
                                          secret_mode_for_poly=secret_mode, 
                                          out=out)
    if multivariate_init is None:
        return None  # Exit if the polynomial is not provided

    # Check if the field is of prime order
    if not isprime(multivariate_init.domain.mod):
        out.print("Sorry, we can only handle prime fields at present.")
        return None  # Exit if the field is non-prime

    commit_hash = poly_commitment_sha256(multivariate_init) if use_commitment else None
    
    dishonest: List[bool] = [False]   # default; will be overwritten

    dishonest_opening: bool = False
    
    if use_commitment:
        if not user_is_prover:
            opening_selection = out.input(
                f"\nChoose a, b, or c: About the opened polynomial g, the prover will "
                f"(a) lie. "
                f"(b) lie with probability 1/2. "
                f"(c) not lie.",
                hidden=secret_mode,
            ).strip().lower()

            if opening_selection == "a":
                dishonest_opening = True
            elif opening_selection == "b":
                dishonest_opening = bool(random.randint(0, 1))
            else:  # c or anything other than a or b
                dishonest_opening = False

        else:
            # Prover will decide later at opening time.
            dishonest_opening = False  # irrelevant for prover-driven opening
            out.print(
                "\nAt the end, the prover will choose what to open with."
                )

    if not user_is_prover:
        dishonesty_selection = out.input(
            f"\nChoose a, b, or c: About the value of H (the sum of g over the Boolean hypercube), the prover will "
            f"(a) lie. "
            f"(b) lie with probability 1/2. "
            f"(c) not lie.",
            hidden=secret_mode,
            ).strip().lower()

        if dishonesty_selection == "a":
            dishonest: List[bool] = [True]
        elif dishonesty_selection == "b":
            dishonest: List[bool] = [bool(random.randint(0, 1))]
        else: # c or anyhing other than a or b
            dishonest: List[bool] = [False]

    # Extract polynomial variables, field, and number of variables
    X = multivariate_init.gens
    v = len(X)
    F = multivariate_init.domain

    # Calculate the true sum by evaluating the polynomial over all Boolean inputs
    H = F(sum(
        multivariate_init.eval({X[k]: b_k for k, b_k in enumerate(b)})
        for b in product([F(0), F(1)], repeat=v)
    ))

    # Initialize prover's claim for sum over Boolean hypercube as the truth
    H_star = H
    # If user is acting as prover:
    if user_is_prover:
        # skip is a boolean and will be set to True.
        # This signifies that initialization step will be shown.
        # It will be skipped next time sum_check_steps is called, so as not to show the same step twice.
        user_H_star, skip = sum_check_steps(g_init=multivariate_init,
                                            H_star=H,
                                            user_is_verifier=user_is_verifier,
                                            user_is_prover=user_is_prover,
                                            use_commitment=use_commitment,
                                            commit_hash=commit_hash,
                                            dishonest_opening=dishonest_opening,
                                            secret_mode=secret_mode,
                                            out=out,)
        user_H_star = int(user_H_star)
        H_star = F.convert(user_H_star)
        if H_star == H:
            dishonest: List[bool] = [False]
        else:
            dishonest: List[bool] = [True]
    if not user_is_prover:
        # See above for explanation of `skip` boolean.
        # In this case, we do not call sum_check_steps here and don't want to skip the initialization when it is called for the first time.
        skip: bool = False
    # If prover is attempting to deceive, choose a different value for H_star
    if not user_is_prover and dishonest[-1]:
        while H_star == H:
            H_star = random_field_element(field=F, user_input=False, out=out,)
    # Call the recursive sum-check protocol
    result = sum_check_recursion(
        g_init=multivariate_init,
        g=multivariate_init,
        H_star=H_star,
        dishonest=dishonest,
        user_is_verifier=user_is_verifier,
        user_is_prover=user_is_prover,
        show_steps=True,
        skip_show_step=skip,
        use_commitment=use_commitment,
        commit_hash=commit_hash,
        dishonest_opening=dishonest_opening,
        secret_mode=secret_mode,
        out=out,
    )
    print_header("Ground truth", level=1, out=out)

    # Truncated display of true g
    X_ = stringify(multivariate_init.gens)
    expr_str = str(multivariate_init.as_expr())
    MAX = 800  # characters

    if len(expr_str) > MAX:
        expr_str = expr_str[:MAX] + " ... (truncated)"

    out.print(f"\nTrue polynomial g:\n\ng({X_}) = {expr_str}")

    # If commitments were used, clarify opening behavior
    if use_commitment:
        out.print("\nCommitment behavior:")
        if dishonest_opening:
            out.print(f"\nProver attempted a {RED}dishonest opening{RESET}.")
        else:
            out.print(f"\nProver opened the {GREEN}correct polynomial{RESET}.")

    # True value of H
    out.print(f"\nTrue value of H := sum g(b) over b in {{0,1}}^{v} is: {int(H)}")
    out.print(f"\nProver claimed that this is equal to: H* = {int(F(H_star))}.")

    if result:
        out.print(f"\nOutcome: V {GREEN}ACCEPTED{RESET}.")
        if F(H_star) == F(H):
            out.print(f"\n{GREEN}This was a TRUE POSITIVE{RESET}: the claim H = H* was correct.")
        else:
            out.print(f"\n{RED}This was a FALSE POSITIVE{RESET}: V accepted a false claim (P got lucky).")
    else:
        out.print(f"\nOutcome: V {RED}REJECTED{RESET}.")
        if F(H_star) == F(H):
            out.print(f"{YELLOW}\nThis was a FALSE NEGATIVE{RESET}: rejection happened despite a true claim (should be rare unless opening failed / transcript inconsistent).")
        else:
            out.print(f"{GREEN}\nThis was a TRUE NEGATIVE{RESET}: V rejected a false claim.")

    out.print("")

    if hasattr(out, "flush"):
        out.flush()
    return result

@count_calls
def sum_check_recursion(
    g_init: Poly,
    g: Poly,
    H_star: Union[int, ModularInteger],
    r: Union[None, List[Union[int, ModularInteger]]] = None,
    dishonest: List[bool] = [False],
    user_is_verifier: bool = False,
    user_is_prover: bool = False,
    show_steps: bool = True,
    skip_show_step: bool = False,
    use_commitment: bool = False,
    commit_hash: Optional[str] = None,
    dishonest_opening: bool = False, 
    secret_mode: Optional[bool] = None,
    *,
    out: TerminalOutput = out,
) -> bool:
    """
    Recursively executes the rounds of the sum-check protocol, verifying each polynomial claim.

    Each recursive call corresponds to one round of the protocol:
    - Reduces the current polynomial by substituting a challenge and summing over the remaining variables.
    - Checks that the claimed univariate polynomial (g*_j) meets the degree bound and sum conditions.
    - If the claim is consistent, generates the next random challenge and proceeds to the next variable.
    - If any inconsistency or degree violation is found, the verifier rejects.

    Args:
        g_init (Poly): The original multivariate polynomial chosen at the start.
        g (Poly): The current polynomial after substituting previously chosen challenges.
        H_star (Union[int, ModularInteger]): The current claimed sum to verify at this recursion step.
        r (Union[None, List[Union[int, ModularInteger]]], optional): The list of challenges chosen so far.
        dishonest (List[bool]): A record of whether dishonesty is being attempted at each step.
        user_is_verifier (bool): True if the user is acting as the verifier.
        user_is_prover (bool): True if the user is acting as the prover.
        show_steps (bool): If True, prints detailed explanations of each step.
        skip_show_step (bool): If True, avoids re-printing certain initialization steps.

    Returns:
        bool: True if the sum-check conditions hold up to this point (and eventually leads to final acceptance if all steps pass),
              False if a violation is found and the verifier rejects.
    """
    # Reset call count if r is empty, else keep the current count
    sum_check_recursion.call_count = 0 if not r else sum_check_recursion.call_count

    # Initialize field and the challenges list if empty
    F = g.domain
    r = r or []

    # Substitute the last element in r into g if r is non-empty
    g = g.subs({g.gens[0]: F(r[-1])}) if r else g

    # # If g is now a constant, perform final verification
    # if g.as_expr().is_constant():
    # The above condition was the wrong condition.
    # Once sum_check_recursion has been called v + 1 times, we're ready for the final verification
    if sum_check_recursion.call_count == len(g_init.gens):
        if show_steps:
            ok = sum_check_steps(g_init=g_init,
                                 g_0_star=g,
                                 H_star=H_star,
                                 r=r,
                                 dishonest=dishonest,
                                 user_is_prover=user_is_prover,
                                 final_check=True,
                                 use_commitment=use_commitment,
                                 commit_hash=commit_hash,
                                 dishonest_opening=dishonest_opening,
                                 secret_mode=secret_mode,
                                 out=out,)
            out.flush()
            if ok is False:
                return False
        # Check if the final evaluated polynomial matches the original claim
        return H_star == F(g_init.eval({g_init.gens[k]: r_ for k, r_ in enumerate(r)}))

    # Define variables for the current polynomial's generators and the number of variables
    X, v = g.gens, len(g.gens)

    # Sum over Boolean values for all variables except the first to obtain g_0
    g_0 = sum(g.subs({X[k + 1]: F(b_k) for k, b_k in enumerate(b)})
              for b in product([F(0), F(1)], repeat=v - 1))

    if dishonest[-1]:
        j = sum_check_recursion.call_count
        try:
            d = g_init.degree(g_init.gens[j])
        except IndexError:
            d = g_init.degree(g_init.gens[j - 1])
        if not user_is_prover:
            g_0_star, roots_of_g_0_star_minus_g_0, _ = dishonest_polynomial_with_boundary_condition(g_0=g_0,max_degree=d, H_star=H_star)
        else:
            g_0_star_suggestion, roots_of_g_0_star_minus_g_0, _ = dishonest_polynomial_with_boundary_condition(g_0=g_0,max_degree=d,H_star=H_star)
    else:
        g_0_star = g_0

    if show_steps and user_is_prover and dishonest[-1]:
        res = sum_check_steps(g_init=g_init,
                                    g_0 = g_0,
                                    g_0_star=g_0_star_suggestion,
                                    roots_of_g_0_star_minus_g_0=roots_of_g_0_star_minus_g_0,
                                    H_star=H_star,
                                    r=r,
                                    dishonest=dishonest,
                                    user_is_verifier=user_is_verifier,
                                    user_is_prover=user_is_prover,
                                    skip=skip_show_step, 
                                    use_commitment=use_commitment,
                                    commit_hash=commit_hash,
                                    dishonest_opening=dishonest_opening,
                                    secret_mode=secret_mode,
                                    out=out,)
        if res is False:
            return False
        assert isinstance(res, Poly)
        g_0_star = res

    elif show_steps:
        ok = sum_check_steps(g_init=g_init,
                        g_0_star=g_0_star,
                        H_star=H_star,
                        r=r,
                        dishonest=dishonest,
                        user_is_verifier=user_is_verifier,
                        user_is_prover=user_is_prover,
                        skip=skip_show_step,
                        use_commitment=use_commitment,
                        commit_hash=commit_hash,
                        dishonest_opening=dishonest_opening,
                        secret_mode=secret_mode,
                        out=out,)
        if ok is False:
            return False

    # Verify degree consistency
    j = sum_check_recursion.call_count
    d = g_0.degree(g_init.gens[j])
    if g_0_star.degree() > d:
        return False  # Reject if degree exceeds the claim

    # Check if the sum of evaluations at 0 and 1 equals H
    if F(H_star) == F(g_0_star.eval({X[0]: F(0)}) + g_0_star.eval({X[0]: F(1)})):
        # Generate a new challenge if verification passed
        verifier_prompt = f"\nAs verifier, choose an element of {F} uniformly at random, independent of any previous choice:" if user_is_verifier else None
        challenge = random_field_element(field=F, user_input=user_is_verifier, custom_message=verifier_prompt, out=out,)
        r.append(challenge)  # Add challenge to the list
        H_star = g_0_star.eval({X[0]: F(challenge)})  # Update H with new evaluation
        if dishonest and H_star == g_0.eval({X[0]: F(challenge)}):
            # Prover has successfuly deceived verifier and verifier will accept if prover gives 'true' polynomials from now on.
            dishonest.append(False)
        # Recurse with the updated values
        return sum_check_recursion(g_init=g_init,
                                   g=g,
                                   H_star=H_star,
                                   r=r,
                                   dishonest=dishonest,
                                   user_is_verifier=user_is_verifier,
                                   user_is_prover=user_is_prover,
                                   use_commitment=use_commitment,
                                   commit_hash=commit_hash,
                                   dishonest_opening=dishonest_opening,
                                   secret_mode=secret_mode,
                                   out=out,)
    else:
        return False  # Reject if sum-check claim fails


def sum_check_steps(
    g_init: Poly,
    g_0: Union[None, Poly] = None,
    g_0_star: Union[None, Poly] = None,
    roots_of_g_0_star_minus_g_0: Union[None, List[Union[int, ModularInteger]]] = None,
    H_star: Union[None, int, ModularInteger] = None,
    r: Union[None, List[Union[int, ModularInteger]]] = None,
    final_check: bool = False,
    dishonest: List[bool] = [False],
    user_is_verifier: bool = False,
    user_is_prover: bool = False,
    skip: bool = False,
    use_commitment: bool = False,
    commit_hash: Optional[str] = None,
    dishonest_opening: bool = False, 
    secret_mode: Optional[bool] = None,
    *,
    out: TerminalOutput = out,
) -> Union[None, bool, Tuple[str, bool], Poly]:
    """
    Displays and guides through the detailed reasoning steps of each round in the sum-check protocol.

    NOTE: This function is intentionally print-heavy and is used for exposition.
    """
    j = sum_check_recursion.call_count  # Current recursion depth
    X, v, F = g_init.gens, len(g_init.gens), g_init.domain
    try:
        d = g_init.degree(g_init.gens[j])
    except IndexError:
        d = g_init.degree(g_init.gens[j - 1])

    b: List[str] = [f"b_{k}" for k in range(v)]
    X_ = stringify(X)
    b_ = stringify(b)
    r = r or []

    # ----------------------------
    # Final verification 
    # ----------------------------
    if final_check and r:
        print_header("\nFinal check", level=2, out=out)

        # "sampled; here r_{v-1} = ..."
        sampled_val = int(F(r[-1]))
        out.print(f"\nV samples an element uniformly at random from {F}; here r_{v - 1} = {sampled_val}.")

        if user_is_prover and len(dishonest) > 1 and dishonest[-2] is True and dishonest[-1] is False:
            out.print(f"\n{RED}=== START INVISIBLE TO VERIFIER ==={RESET}")
            out.print(
                f"\n{RED}YOU HAVE SUCCESSFULLY DECEIVED THE VERIFIER: "
                f"g*_{j-1}({int(r[-1])}) = g_{j-1}({int(r[-1])}){RESET}"
            )
            out.print(f"\n{RED}=== END INVISIBLE TO VERIFIER ==={RESET}")

        r_ = stringify(r)

        LHS = f"g*_{j - 1}({to_string(r[j - 1])})"
        RHS = f"g({r_})"
        subs = {g_init.gens[k]: r[k] for k in range(len(r))}

        # Always the committed polynomial's value (conceptually what PCS opening proves)
        g_comm_at_r = int(F(g_init.eval(subs)))

        out.print(f"\nIf all of P's claims are true, then\n\n{LHS} = {RHS}.")
        out.print(f"\nV computes\n\n{LHS} = {H_star}.")

        g_open_at_r: Optional[int] = None

        if use_commitment and commit_hash is not None:
            out.print("\nP now reveals g (this is the 'opening' step in our toy commitment demo).")
            out.print(f"\n{YELLOW}CAVEAT (real protocols): "
                f"In practice, P does *not* reveal g. Instead, P commits to g using a binding "
                f"(and typically hiding) polynomial commitment scheme and later proves that "
                f"g(r) equals the claimed value using a succinct opening proof.{RESET}"
                )
            
            g_open = g_init

            if user_is_prover:
                # Let prover choose what to open with (honest or not).
                choice = out.input(
                    "\nPress ENTER to open honestly, or type 'x' to enter a different polynomial to open: ",
                    hidden=secret_mode,
                ).strip().lower()

                if choice == "x":
                    # Prover supplies an opening polynomial (can be dishonest).
                    g_open = choose_polynomial(
                        field=F,
                        custom_message="\nEnter the polynomial you want to open with:",
                        variable_names=[str(x) for x in g_init.gens],
                        prompt_for_k=False,
                        secret_mode_for_poly=bool(secret_mode),  # hide typing if you want
                        out=out,
                    )
                    if g_open is None:
                        out.print(f"\nV {RED}REJECTS{RESET}: no polynomial provided at opening.")
                        return False
            else:
                # Non-prover mode: keep your scripted dishonesty.
                if dishonest_opening:
                    g_open = g_init + F(1)


            # if dishonest_opening:
            #     g_open = g_init + F(1)   # same vars/field, changes constant term

            # Optional: if you want to hide that prover chose to lie until the end:
            if not secret_mode and dishonest_opening:
                out.print(f"\n{RED}In this run, P will attempt to open with a different polynomial.{RESET}")

            if dishonest_opening:
                out.print(f"\nRevealed polynomial:\n\n{RED}g({X_}) = {g_open.as_expr()}{RESET}.")
            else:
                out.print(f"\nRevealed polynomial:\n\n{GREEN}g({X_}) = {g_open.as_expr()}{RESET}.")

            out.print("\nV recomputes the hash of the revealed polynomial and checks it matches the original commitment.")

            recomputed = poly_commitment_sha256(g_open)
            ok = (recomputed == commit_hash)

            out.print(f"\nCommitment check: {GREEN}PASS{RESET}" if ok else f"\nCommitment check: {RED}FAIL{RESET}")
            out.print(f"\nOriginal hash:   {commit_hash}")
            if ok:
                out.print(f"Recomputed hash: {GREEN}{recomputed}{RESET}")             
            if not ok:
                out.print(f"Recomputed hash: {RED}{recomputed}{RESET}")
                out.print(f"\nV {RED}REJECTS{RESET}: the revealed polynomial does not match the original commitment.")
                return False
            
            # Demo: since g is revealed, V can evaluate the opened polynomial:
            g_open_at_r = int(F(g_open.eval(subs)))
            out.print(f"\nFor the purposes of this demo (see caveat above): since g is revealed, V can evaluate it directly:\n\n{RHS} = {g_open_at_r}.")

        else:
            out.print(
                f"\nFinally, V queries an oracle for g at r to obtain\n\n{RHS} = {g_comm_at_r}."
            )

        check_val = g_open_at_r if (use_commitment and commit_hash is not None) else g_comm_at_r
        if F.convert(H_star) == F.convert(check_val):
            out.print(f"\nP passes the final verification, and V {GREEN}ACCEPTS{RESET} P's initial claim that H = H*.")
            H = F(sum(g_init.eval({X[k]: b_k for k, b_k in enumerate(bb)})
                      for bb in product([F(0), F(1)], repeat=v)))
            if dishonest and dishonest[0]:
                out.print(f"\n{RED}P HAS DECEIVED V: THE TRUE VALUE OF H IS IN FACT {int(H)}.{RESET}")
                # Meta line: highlight that acceptance was a low-probability event.
                out.print(
                    f"\n{YELLOW}P GOT LUCKY:{RESET} the final random check happened not to expose the inconsistency "
                    f"this time. Over a large field, this success event should be rare."
                )
        else:
            out.print(f"\nP fails the final verification, and V {RED}REJECTS{RESET} P's initial claim that H = H*.")
            out.print(
                f"\n{YELLOW}Interpretation: P can often keep the transcript locally consistent round-by-round, "
                f"but the final random evaluation ties the transcript to the true g and typically catches a lie.\n{RESET}"
            )
        return

    # ----------------------------
    # Initialization (only when r empty and not skipped)
    # ----------------------------
    if not r and not skip:
        print_header("Initialization", out=out)

        degs = [int(g_init.degree(gen)) for gen in g_init.gens]

        if secret_mode is False or secret_mode is None: 
            out.print(f"\nLet\n\ng({X_}) = {g_init.as_expr()}.")
            out.print(
            f"\n{YELLOW}We display g for the reader. In the protocol, V does not receive g explicitly;{RESET} "
            f"{YELLOW}V only has oracle/committed evaluation access to g.{RESET}"
            ) 
            if use_commitment and commit_hash is not None:
                out.print(f"\nCommitment hash (binding commitment to g): {commit_hash}")
            out.print(f"\nFor soundness, V relies on degree bounds per variable (this is public information): {degs}.")

        if secret_mode and use_commitment and commit_hash is not None:
            out.print(f"\nV receives a commitment to g (not g itself).")
            out.print(f"\nCommitment hash: {commit_hash}")
            out.print(f"\nFor soundness, V relies on degree bounds per variable (this is public information): {degs}.")
            out.print(f"\nV proceeds knowing only the field, arity, degree bounds, and a binding commitment to g.")     

        elif secret_mode:
            out.print(f"\nFor soundness, V relies on degree bounds per variable (this is public information): {degs}.")                

        out.print(f"\nLet\n\nH := sum g({b_}) over ({b_}) in {{0,1}}^{v}.")

        if user_is_prover:
            if secret_mode is True:
                user_H_star = out.input(f"\nAs prover, enter your claim H* for the value of H:", hidden=secret_mode)
            else:
                user_H_star = out.input(f"\nAs prover, enter your claim H* for the value of H (the true value is {int(H_star)}):")
            skip = True
            return user_H_star, skip
        else:
            # Fix "P claims that and H..."
            out.print(f"\nP claims that H = H*, where H* = {int(F(H_star))}.")
            if dishonest and dishonest[0] and not(secret_mode is True):
                out.print(
                    f"\n{RED}In this run, P's initial claim H* is intentionally false, and P will try to get away with it.{RESET}")

    # ----------------------------
    # If we are past initialization, report the last sampled challenge r_{j-1}
    # ----------------------------
    if r:
        sampled_val = int(F(r[-1]))
        out.print(f"\nV samples an element uniformly at random from {F}; here r_{j - 1} = {sampled_val}.")
        if user_is_prover and len(dishonest) > 1 and dishonest[-2] is True and dishonest[-1] is False:
            out.print(f"\n{RED}=== START INVISIBLE TO VERIFIER ==={RESET}")
            out.print(
                f"\n{RED}YOU HAVE SUCCESSFULLY DECEIVED THE VERIFIER: "
                f"g*_{j}({int(r[-1])}) = g_{j}({int(r[-1])}){RESET}"
            )
            out.print(f"\n{RED}From now on, the g*_j will automatically be set to g_j.{RESET}")
            out.print(f"\n{RED}=== END INVISIBLE TO VERIFIER ==={RESET}")

    # ----------------------------
    # Round header and definition of g_j
    # ----------------------------
    print_header(f"Round {j}", out=out)
    mixed_input = stringify(r[:j], [X[j]], b[j + 1:])
    if v - 1 - j > 0:
        out.print(
            f"\nLet\n\ng_{j}({str(X[j])}) := sum g({mixed_input}) over ({stringify(b[j + 1:])}) in {{0,1}}^{v - 1 - j}"
        )
    else:
        out.print(f"\nLet\n\ng_{j}({str(X[j])}) := g({mixed_input})")

    # ----------------------------
    # Prover-hidden advice block 
    # ----------------------------
    if user_is_prover and dishonest[-1]:
        out.print(f"\n{RED}=== START INVISIBLE TO VERIFIER ==={RESET}")
        out.print(f"\nIn fact, \n\n{GREEN}g_{j}({str(X[j])}) = {g_0.as_expr()}{RESET}.")
        out.print(
            f"\nYour goal is to convince V that the polynomial g*_{j} you're about to send is g_{j} (a false claim)."
        )
        out.print(f"\nV will check that:")
        out.print(f"\n(a) deg(g*_{j}) ≤ deg_{j}(g) = {d}")
        out.print(f"\n(b) g*_{j}(0) + g*_{j}(1) = {int(H_star)}")
        out.print(
            f"\nIf it turns out that"
            f"\n(c) g*_{j}(r_{j}) = g_{j}(r_{j}), where r_{j} is the random element of {F} that V will sample next,"
        )
        out.print(
            f"\nthen V will accept your original claim if you send the 'true' polynomials g*_k = g_k in subsequent rounds."
        )
        out.print(
            f"\nAs you don't know what r_{j} will be, the best you can do is choose g*_{j} satisfying (a) and (b) such that"
        )
        out.print(
            f"\n(d) g*_{j} - g_{j} has as many roots as possible in {F} "
            f"(there can be at most deg(g*_{j} - g_{j}) ≤ {d})."
        )
        if g_0_star and roots_of_g_0_star_minus_g_0:
            roots = [int(z) for z in roots_of_g_0_star_minus_g_0]
            print_roots = ", ".join([f"{YELLOW}{str(z)}{RESET}" for z in roots])
            out.print(
                f"\nFor example, if{RED}\n\ng*_{j}({str(X[j])}) = {g_0_star.as_expr()}{RESET},\n\nthen (a) and (b) hold, "
                f"and the roots (in {F}) of g*_{j} - g_{j} are: {print_roots}."
            )
            out.print(f"\nThe probability of V sampling one of these roots as the next challenge is "
                  f"{len(roots)}/{F.mod} = {len(roots)/F.mod:.2f}.")
            out.print(
                f"\nIn general, by Schwartz-Zippel, this probability cannot exceed "
                f"deg_{j}(g)/#F = {g_init.degree(X[j])}/{F.mod} = {g_init.degree(X[j])/F.mod:.2f}."
            )
            out.print(f"\nIn this event, V will ultimately accept the original false claim for the value of H.")
            out.print(
                f"\nEven if this does not happen, as long as (a) and (b) hold, V will accept in this round "
                f"and we will have another chance to pull off the deception."
            )
        elif g_0_star:
            out.print(
                f"\nFor example, if\n\n{RED}g*_{j}({str(X[j])}) = {g_0_star.as_expr()}{RESET},\n\nthen (a) and (b) hold, "
                f"but g*_{j} - g_{j} has no roots in {F}."
            )
            out.print(
                f"\nWe are unable to suggest a polynomial satisfying (a) and (b) such that g*_{j} - g_{j} has a root "
                f"in {F}—perhaps no such polynomial exists."
            )
            out.print(
                f"\nEven if no such polynomial exists, as long as (a) and (b) hold, V will accept in this round "
                f"and we will have another chance to pull off the deception."
            )
        else:
            out.print(f"\nWe are unable to suggest a polynomial satisfying (a) and (b)—perhaps no such polynomial exists.")
            out.print(f"\nIf no such polynomial exists, we've been caught in a lie and V will reject immediately.")

        prompt = f"\nEnter your g*_{j}({str(X[j])}):"
        g_0_star = choose_polynomial(field=F, custom_message=prompt, variable_names=[f"X_{j}"], prompt_for_k=False, out=out)
        out.print(f"\n{RED}=== END INVISIBLE TO VERIFIER ==={RESET}")

    # ----------------------------
    # Prover's claim in this round
    # ----------------------------
    out.print(
        f"\nP claims that g_{j}({str(X[j])}) = g*_{j}({str(X[j])}), where \n\ng*_{j}({str(X[j])}) = {g_0_star.as_expr()}."
    )

    # Equality check to be verified in this round
    if j == 0:
        LHS = "H*"
    else:
        LHS = f"g*_{j - 1}({to_string(r[j - 1])})"
    RHS = f"g*_{j}(0) + g*_{j}(1)"
    out.print(f"\nIf P's last two claims are true, then deg(g*_{j}) ≤ deg_{j}(g) and {LHS} = {RHS}.")

    # Degree bound check (soundness-relevant)
    if g_0_star.degree() <= d:
        out.print(
            f"\nV checks the degree bound: deg(g*_{j}) = {g_0_star.degree()} ≤ deg_{j}(g) = {d}."
        )
    else:
        out.print(f"\nV finds that deg(g*_{j}) = {g_0_star.degree()} > deg_{j}(g) = {d}.")
        out.print(f"\nV {RED}REJECTS{RESET} (degree bound violated). The protocol terminates.")
        return False         

    # Consistency check for the sum relation
    if LHS == "H*":
        out.print(f"\nP's claim is that H = H* = {int(F(H_star))}. V proceeds to compute:\n")
        computed_rhs = int(F(g_0_star.eval(F(0)) + g_0_star.eval(F(1))))
        out.print(f"{RHS} = {computed_rhs}")
        out.print(f"\nThis is easy because g*_{j} is given explicitly, so V can evaluate it at 0 and 1.")
    else:
        out.print("\nV proceeds to compute:\n")
        computed_rhs = int(F(g_0_star.eval(F(0)) + g_0_star.eval(F(1))))
        display_aligned(f"{LHS} = {int(F(H_star))}", f"{RHS} = {computed_rhs}", out=out)
        out.print(
            f"\nThis is easy because both claimed polynomials are given explicitly, "
            f"and V only needs univariate evaluations at 0 and 1."
        )

    if F(H_star) != F(g_0_star.eval(F(0)) + g_0_star.eval(F(1))):
        out.print(f"\nV {RED}REJECTS{RESET} upon exposing an inconsistency in P's claims. The protocol terminates.")
        return False
    else:
        out.print("\nV sees that P's claims so far are consistent.")

    if user_is_prover and dishonest:
        return g_0_star


# Helper functions

def stringify(*args: Union[Tuple[Symbol], List[Union[str, int, 'ModularInteger']]]) -> str:
    """
    Converts a list of coordinates (tuples or lists) into a single formatted string.

    Args:
        *args (Union[Tuple[Symbol], List[Union[str, int, ModularInteger]]]):
              Variable-length arguments of tuples or lists containing coordinates.

    Returns:
        str: A comma-separated string representation of the coordinates.
    """
    result = []
    for coordinates in args:
        if not coordinates:
            continue  # Skip empty coordinates

        # Convert each element in the coordinates to a string
        coordinates_str = [to_string(c) for c in coordinates]

        # Join the coordinates with commas and add to the result list
        result.append(', '.join(coordinates_str))

    # Return all coordinate strings joined with commas
    return ', '.join(result)


def to_string(c):
    """
    Converts an element to its string representation based on its type.

    Args:
        c: The element to be converted (could be a Symbol, ModularInteger, or int).

    Returns:
        str: The string representation of the element.
    """
    if isinstance(c, Symbol):
        return str(c)  # Convert Symbol to string
    if isinstance(c, ModularInteger):
        return str(int(c))  # Convert ModularInteger to int, then to string
    if isinstance(c, int):
        return str(c)  # Convert int to string
    return str(c)  # Default conversion to string


"""ANCILLARY FUNCTIONS"""

def random_field_element(
    field: FiniteField,
    user_input: bool = False,
    custom_message: Union[None, str] = None,
    max_attempts: Optional[int] = None,
    *,
    out: TerminalOutput = out,
) -> Optional[ModularInteger]:
    """
        Generates a random element from the finite field.

        Args:
            field (FiniteField): The finite field GF(p).
            user_input (bool): If True, prompts the user to input the element.
            custom_message (Union[None, str], optional): A custom message to display to the user when prompting
                                                         for input. Defaults to None, in which case a default
                                                         message is used.
            max_attempts (Optional[int], optional): Maximum number of attempts for user input. Defaults to None,
                                                    meaning unlimited attempts.

        Returns:
            Optional[ModularInteger]: A random field element if successful, or None if the user cancels or
                                      exceeds the maximum number of attempts.

        Notes:
            - If `user_input` is True, the function will prompt the user to input a field element.
            - If `custom_message` is provided, it will override the default message displayed to the user.
            - If `user_input` is False, the function generates a random element from the finite field.
        """
    if user_input:
        user_response = input_random_field_element(custom_message=custom_message, max_attempts=max_attempts, out=out,)
        if isinstance(user_response, int):
            return field(user_response)
        else:
            return None
    # Generate a random integer between 0 and p - 1
    return field(random.randint(0, field.mod - 1))


def input_random_field_element(custom_message: Union[None, str] = None, max_attempts: Optional[int] = None, *, out: TerminalOutput = out) -> Optional[int]:
    """
        Prompts the user to input an integer element from the field.

        Args:
            custom_message (Union[None, str], optional): A custom message to display when prompting the user for input.
                                                         Defaults to a standard prompt if not provided.
            max_attempts (Optional[int], optional): Maximum number of attempts allowed for input. If None, unlimited
                                                    attempts are allowed.

        Returns:
            Optional[int]: The integer entered by the user if valid, or None if the user cancels or the maximum
                           number of attempts is exhausted.

        Notes:
            - If `custom_message` is provided, it overrides the default prompt message shown to the user.
            - If the user inputs `c`, the function cancels and returns None.
            - If `max_attempts` is specified, the user is limited to the given number of attempts.
            - When the maximum number of attempts is reached without valid input, the function returns None.
        """
    attempts = 0 if max_attempts is not None else None
    prompt = custom_message if custom_message is not None else "\nEnter c to cancel or select element uniformly at random from field, independent of any previous selection:"

    while attempts is None or (attempts != max_attempts):
        response = out.input(prompt)
        if response == 'c':
            return None
        try:
            response_int = int(response)
            return response_int
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

"""
START: FUNCTIONS RELATED TO ROLE OF PROVER, WHETHER DISHONEST OR HONEST
"""

def dishonest_polynomial_with_boundary_condition(g_0: Poly,
                                                 H_star: Union[int, ModularInteger],
                                                 max_degree: Union[None, int] = None,
                                                 random_roots: bool = False,
                                                 timeout: Optional[int] = None,) -> Tuple[Union[None, Poly], Union[None, List[Union[int, ModularInteger]]], Union[None, bool]]:
    """
    Constructs a modified polynomial g_0^* recursively under a boundary condition.

    This function modifies the polynomial g_0 such that:
        - g_0^* is distinct from g_0,
        - g_0^*(0) + g_0^*(1) = H_star (the `boundary condition`),
        - deg(g_0^*) <= deg(g_0) or is limited by `max_degree` if specified.

    If H_star = g_0(0) + g_0(1), the boundary condition is trivial, and the function calls
    `dishonest_polynomial_no_boundary_conditions` to construct g_0^* without the boundary constraint.

    Degenerate cases are handled:
        - If g_0 is constant or the field characteristic is 2, specific logic ensures that
          g_0^*(0) + g_0^*(1) = H_star while respecting field constraints.
        - If deg(g_0) = 0, g_0^* is constructed as a constant polynomial.
        - If deg(g_0) = 1, g_0^* is linear or constant, satisfying the required sum.

    Args:
        g_0 (Poly): A univariate polynomial defined over a prime field.
        H_star (Union[int, ModularInteger]): The target sum g_0^*(0) + g_0^*(1).
        max_degree (Union[None, int], optional): The maximum allowable degree for the constructed polynomial.
                                                 Defaults to None.
        random_roots (bool, optional): Whether to randomly select roots in the construction
                                       (used when H_star = g_0(0) + g_0(1)).
        timeout (Optional[int], optional): Maximum allowed time in seconds for constructing g_0^*.
                                           If exceeded, the function stops the process early and
                                           returns a timeout flag.

    Returns:
        Tuple[Union[None, Poly], Union[None, List[Union[int, ModularInteger]]], Union[None, bool]]:
            - The first element is the constructed polynomial (Poly) or None if no polynomial can be constructed.
            - The second element is the list of roots (as integers or modular integers) or None if no roots are found.
            - The third element is a boolean flag indicating whether a timeout occurred (True if timeout, None otherwise).

    Raises:
        ValueError: If g_0.domain is not a prime field.

    Notes:
        - If H_star = g_0(0) + g_0(1), the function reverts to `dishonest_polynomial_no_boundary_conditions`.
        - The recursive construction starts with alpha_0 = 0 and iteratively computes alpha_k
          to maximize the number of distinct roots, stopping early if no further roots can be added
          or the timeout is exceeded.
        - For fields of characteristic 2, the construction accounts for field-specific behavior
          and ensures validity even in edge cases.
    """
    # Ensure the field is a prime field
    field = g_0.domain
    if not isprime(field.mod):
        raise ValueError("g_0 must be defined over a prime field.")

    # Compute H as a field element
    H = field.convert(g_0.eval(0) + g_0.eval(1))
    # Convert H^* to a field element (if it isn't already)
    H_star = field.convert(H_star)
    # If H_star equals H, there is effectively no boundary condition
    if H_star == H:
        g_0_star, roots_of_g_0_star_minus_g = dishonest_polynomial_no_boundary_conditions(g=g_0, max_degree=max_degree, random_roots=random_roots)
        return g_0_star, roots_of_g_0_star_minus_g, None

    # Compute the difference delta in the field
    delta = H_star - H
    # Field size
    p = field.mod
    # d will be max potential degree of output g_0_star: cannot exceed p - 1 but otherwise should be as large as deg(g_0) (or max_degree, if specified) if possible
    if max_degree:
        assert g_0.degree() <= max_degree, "max_degree must be greater than or equal to deg(g_0)"
        d = min(max_degree, p - 1)
    else:
        d = min(g_0.degree(), p - 1)
    # Deal with some degenerate cases
    if d < 0: # g_0 is the zero polynomial and g_0_star must also be. But we require g_0_star to be distinct from g_0
        return None, None, None
    if d == 0: # g_0 is constant (possibly zero) and g_0_start must also be---a different constant.
        if p > 2:
            g_0_star = Poly(H_star / field.convert(2), g_0.gens[0], domain=field)  # So that 2*g_0_star = H_star
            if g_0_star == g_0:
                return None, None, None
            return g_0_star, [], None
        # Now we're in the case p = 2
        if H_star == field.one:
            return None, None, None  # For any constant polynomial g_0_star over Z/2Z, 2*g_0_star = 0 mod 2
        # H_star == field.zero
        g_0_star = Poly(field.zero, g_0.gens[0], domain=field)
        if g_0_star == g_0:
            return None, None, None
        return g_0_star, [], None
    if d == 1: # g_0_star must be constant or linear and satisfy g_0_star(0) + g_0_star(1) = H_star
        if p == 2: # g_0 is either x or x + 1 or 0 or 1
            if g_0.degree() == 1: # g_0 is either x or x + 1
                g_0_star = Poly(g_0 + 1, g_0.gens[0], domain=field)
                if field.convert(g_0_star.eval(0) + g_0_star.eval(1)) == H_star:
                    return g_0_star, [], None
                # Otherwise, g_0_star must be constant
                if H_star == field.one:
                    return None, None, None
                g_0_star = field.zero
                return g_0_star, [g_0.eval(0)], None
            # g_0 is 0 or 1
            if H_star == field.zero:
                g_0_star = Poly(0, g_0.gens[0], domain=field)
                if g_0_star == g_0:
                    return None, None, None
                return g_0_star, [], None
            # H_star == field.one
            g_0_star = Poly(g_0.gens[0], g_0.gens[0], domain=field)
            return g_0_star, [g_0.eval(0)], None
        # Now we're in the case p > 2
        # We'll return a linear polynomial g_0_star = a_star*x + b_star
        # We have to ensure g_0_star(0) + g_0_star(1) = a_star + 2*b_star = H_star
        # We'll choose b_star = b + 1 where g_0(x) = a*x + b. This guarantees g_0_star is different from g_0
        b_star = field.convert(g_0.eval(0) + 1)
        a_star = field.convert(H_star - 2*b_star)
        g_0_star = Poly(a_star*g_0.gens[0] + b_star, g_0.gens[0], domain=field)
        delta_a = a_star - g_0.nth(1)
        if delta_a%p != 0:
            roots = [1/field.convert(-delta_a)]
        else:
            roots = None
        return g_0_star, roots, None
    # Helper function to compute alpha_k
    def compute_alpha_k(alpha_list):
        """Compute the next alpha_k."""
        prod_0 = field.convert(1)
        prod_1 = field.convert(1)
        for alpha in alpha_list:
            prod_0 *= (field.convert(0) - alpha)
            prod_1 *= (field.convert(1) - alpha)
        denominator = prod_0 + prod_1
        if denominator == 0:
            return None  # Cannot compute alpha_k due to zero denominator
        numerator = delta - prod_1
        alpha_k = - numerator / denominator
        return field.convert(alpha_k)

    # Recursive construction of sets of alpha
    best_alpha_list = []  # Track the best (longest) list of alpha
    out_of_time = False
    if timeout is not None:
        tic = time.time()  # Start timing
    for alpha_0 in range(p):  # Iterate over possible starting points for alpha_0
        alpha_list = [field.convert(alpha_0)]  # Start with alpha_0
        for _ in range(d - 1):  # Limit the length to d
            alpha_k = compute_alpha_k(alpha_list)
            if alpha_k is None or alpha_k in alpha_list:
                break  # Stop if alpha_k is invalid or not distinct
            alpha_list.append(alpha_k)
        if len(alpha_list) > len(best_alpha_list):
            best_alpha_list = alpha_list
        if timeout is not None:
            toc = time.time()  # Current time
            out_of_time: bool = toc - tic > timeout
        else:
            out_of_time: bool = False
        if len(best_alpha_list) == d or out_of_time:
            break  # Stop early if we find a list of length d, or if timeout is exceeded
    if not best_alpha_list:
        if timeout: # found nothing useful in the time given
            return None, None, out_of_time
        else: # found nothing useful
            return None, None, None
    # Construct the correction term using the best alpha list
    correction_term = Poly(1, g_0.gens[0], domain=field)
    for alpha in best_alpha_list:
        correction_term *= Poly(g_0.gens[0] - alpha, g_0.gens[0], domain=field)

    # Construct g_0^*
    g_0_star = g_0 + correction_term
    if timeout:
        return g_0_star, best_alpha_list, out_of_time
    return g_0_star, best_alpha_list, None

def dishonest_polynomial_no_boundary_conditions(g: Poly, max_degree: Union[None, int] = None, random_roots: bool = False) -> Tuple[Union[None, Poly], Union[None, List[Union[int, Any]]]]:
    """
    Given a univariate polynomial g over a finite field F (expected to be GF(p)),
    construct a new polynomial g*(x) = g(x) + h(x) satisfying the constraints:

    1. If deg(g) > 1:
       Let d = min(g.degree(), p-1, max_degree).
       If random_roots=False:
           h(x) = x*(x-1)*∏(x - i) for i=2,...,d-1.
       If random_roots=True:
           h(x) = x*(x-1)*∏(x - α_i), where α_i are d-2 distinct random elements from {2, 3, ..., p-1}.
       This ensures h(0)+h(1)=0 and gives d distinct roots.
       Returns a tuple (g(x) + h(x), list of chosen roots).

    2. If deg(g) = 1 and char(F) ≠ 2:
       Returns a tuple (g(x) + (x - 1/2), [1/2]), where 1/2 is the inverse of 2 mod p.

    3. If deg(g) = 1 and char(F) = 2:
       Then g(x) is either x or x+1 (over GF(2)).
       - If g(x)=x, returns (Poly(1), [1]) (constant polynomial 1).
       - If g(x)=x+1, returns (Poly(0), [1]) (constant polynomial 0).

    4. If deg(g) = 0, returns (None, None).

    Checks if F is a prime field by verifying that p is prime.

    Parameters
    ----------
    g : Poly
        Univariate polynomial over GF(p).
    max_degree : Union[None, int], optional
        The maximum allowable degree for the constructed polynomial. If provided, it constrains
        the degree of h(x). Defaults to None.
    random_roots : bool, optional
        If True and deg(g) > 1, choose the d-2 elements randomly
        from {2, 3, ..., p-1}. If False, use the sequence 2,...,d-1.

    Returns
    -------
    Tuple[Union[None, Poly], Union[None, List[Union[int, Any]]]]
        A tuple where:
        - The first element is either the modified polynomial g* (Poly) or None.
        - The second element is either the list, as integers or modular integers, of field elements r such that g^*_0(r) = g_0(r) (i.e. the roots of g^*_0 - g_0), or None.

    Raises
    ------
    ValueError
        If F is not a prime field or if conditions are not met.
    """
    if len(g.gens) != 1:
        raise ValueError("Polynomial g must be univariate.")

    F = g.domain

    # Check if F is a prime field by checking if characteristic is prime.
    if F.has_CharacteristicZero:
        raise ValueError("We need a finite prime field. Characteristic zero not allowed.")

    p = F.mod
    if not isprime(p):
        raise ValueError(
            "We are using SymPy and therefore can only handle polynomials over prime fields."
        )

    X = g.gens[0]
    # Set d = min(g.degree(), p - 1)
    if max_degree is not None:
        d = min(max_degree, p - 1)
    else:
        d = min(g.degree(), p - 1)

    if d == 0:
        # deg(g)=0 means g is constant. Return None.
        return None, None

    if d == 1:
        # If deg(g)=1, handle char(F)=p cases
        if p != 2:
            # Need 1/2 in GF(p). Using Fermat's little theorem: 1/2 = 2^(p-2) mod p
            inv_2 = F.convert(pow(2, p - 2, p))
            # h(x) = x - inv_2
            h = Poly(X - inv_2, X, domain=F)
            return g + h, [inv_2]
        else:
            # p=2, g(x)=x or g(x)=x+1
            coeffs = g.all_coeffs()
            # For deg=1: g(x)=a*x+b, coeffs=[a,b]
            a, b = coeffs[0], coeffs[1]
            if a == F.one and b == F.zero:
                # g(x)=x, return constant polynomial 1
                return Poly(F.one, X, domain=F), [F.one]
            elif a == F.one and b == F.one:
                # g(x)=x+1, return constant polynomial 0
                return Poly(F.zero, X, domain=F), [F.one]
            else:
                raise ValueError("In characteristic 2 and deg=1, g(x) should be x or x+1.")

    if d > 1:
        # h(x) = x*(x-1)*∏(x - i) for i=2,...,d-1
        # If random_roots=True, pick d-2 distinct random elements from {2,...,p-1}.
        if d == 2:
            h = Poly(X * (X - F.one), X, domain=F)
            alphas = [F.zero, F.one]
        else:
            # We need d-2 elements. If random_roots is False, we use 2,...,d-1.
            # If random_roots is True, we pick them at random from {2,...,p-1}.
            if random_roots:
                candidates = list(range(2, p))
                random.shuffle(candidates)
                chosen_ints = candidates[:(d - 2)]  # d-2 elements
            else:
                chosen_ints = list(range(2, d))

            alphas = [F.convert(i) for i in chosen_ints]
            h = Poly(X * (X - F.one), X, domain=F)
            for alpha in alphas:
                h *= Poly(X - alpha, X, domain=F)

        return g + h, alphas
    
    return None, None


"""
END: FUNCTIONS RELATED TO ROLE OF PROVER, WHETHER DISHONEST OR HONEST
"""

"""
START: TOY HASH COMMITMENT FUNCTIONS
"""

def poly_commitment_sha256(g: Poly) -> str:
    """
    Toy commitment: SHA-256 hash of a canonical serialization of a SymPy Poly.

    Binding here is only "demo binding":
    - if someone later changes g, the hash will (almost surely) change.
    - not hiding (hash is not a commitment scheme).
    """
    F = g.domain
    p = int(getattr(F, "mod", 0) or 0)
    gens = tuple(str(x) for x in g.gens)

    # Canonical term list: (monom tuple, coeff int) sorted lexicographically
    # Poly.terms() gives [((e0,...,ev-1), coeff), ...]
    terms: list[Tuple[Tuple[int, ...], int]] = []
    for monom, coeff in g.terms():
        try:
            c = int(coeff)
        except Exception:
            c = int(F.convert(coeff))
        terms.append((tuple(int(e) for e in monom), c))

    terms.sort()

    # Degree bounds per variable (what V "knows")
    degs = tuple(int(g.degree(gen)) for gen in g.gens)

    payload = repr((p, gens, degs, terms)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

"""
END: TOY HASH COMMITMENT FUNCTIONS
"""