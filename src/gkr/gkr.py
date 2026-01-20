from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Sequence

from .utils.display import TerminalOutput, out as default_out, print_header, RED, GREEN, YELLOW, BLUE, RESET
from .utils.polynomials import random_field_element

from .ac import ArithmeticCircuit, replace_symbols_in_polynomial
from .sumcheck import sum_check_recursion

from typing import Optional, Tuple, Union
import random
import numpy as np
from sympy import Poly, Symbol, symbols
from sympy.polys.domains import GF, Domain
from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.modularinteger import ModularInteger

def _label_bitlength_from_W_dict(ac, i: int) -> int:
    # keys are tuples like (0,1,0) of length s_i, or () for s_i=0
    k = next(iter(ac.W_dict[i].keys()))
    return len(k)


def _eval_poly_at_point(
    poly: Poly,
    point: List[Union[int, ModularInteger]],
    *,
    field: FiniteField,
) -> ModularInteger:
    """
    Evaluate poly at a point (list of coords) in the order poly.gens.
    """
    if len(poly.gens) != len(point):
        raise ValueError(f"Poly has {len(poly.gens)} gens but point has length {len(point)}.")
    subs = {poly.gens[i]: point[i] for i in range(len(point))}
    return field.convert(poly.eval(subs))


def gkr(
    ac,
    *,
    user_is_verifier: bool = False,
    user_is_prover: bool = False,
    dishonest: bool = False,
    show_steps: bool = True,
    secret_mode: Optional[bool] = None,
    out: TerminalOutput = default_out,
):
    F: FiniteField = ac.field
    d: int = ac.depth  # layers 0..d

    print_header("GKR", level=1, out=out)
    out.print(f"\nField: {F}")
    out.print(f"Layers: 0..{d} (0 = output, {d} = input)")

    # -----------------------
    # Layer 0: output check
    # -----------------------
    s0 = _label_bitlength_from_W_dict(ac, 0)
    out.print(f"\n\nLAYER 0: OUTPUT CHECK\n")

    # sample rho_0 in F^{s0}
    # (use whatever random-field-element helper you settled on elsewhere)
    rho0: Tuple[Union[int, ModularInteger], ...] = tuple(random_field_element(field=F) for _ in range(s0))

    out.print(f"Verifier samples rho_0 uniformly at random from F^{s0}.")
    out.print(f"Here rho_0 = ({', '.join(str(int(x)) for x in rho0)}).")

    out.print("\nClaim being checked: W̃_0(rho_0) = W̃_0^*(rho_0).")
    out.print("In this demo run, we take W̃_0^* := W̃_0 (honest prover).")

    # beta_0 := W̃_0(rho0)
    subs0 = {ac.tilde_W[0].gens[j]: rho0[j] for j in range(s0)} if s0 > 0 else {}
    beta: Union[int, ModularInteger] = F.convert(ac.tilde_W[0].eval(subs0))
    out.print(f"\nComputed β_0 := W̃_0(rho_0) = {int(beta)}.")

    # -----------------------
    # Inductive layers: i = 0..d-1
    # -----------------------
    rho = rho0
    for i in range(0, d):
        out.print(f"\n\nLAYER {i}: REDUCE TO LAYER {i+1}\n")
        res = gkr_layer_transition(
            ac=ac,
            i=i,
            rho_i=rho,
            beta_i=beta,
            honest_prover=not dishonest,          # if dishonest=True, treat as not honest
            dishonest_folding=dishonest,          # for now: dishonest => use cheating helper in folding
            out=out,
        )

        if not res.ok:
            out.print(f"\n{RED}Protocol failed at layer {i}.{RESET}")
            return False

        rho, beta = res.rho_next, res.beta_next

    # -----------------------
    # Final check at layer d (input layer)
    # -----------------------
    out.print(f"\n\nFINAL CHECK (LAYER {d})\n")
    # TODO: compute W̃_d(rho_d) directly from public inputs (for your constant-input demo, you can just use ac.tilde_W[d])
    subsd = {ac.tilde_W[d].gens[j]: rho[j] for j in range(len(rho))} if len(rho) > 0 else {}
    beta_true = F.convert(ac.tilde_W[d].eval(subsd))

    if F.convert(beta) == F.convert(beta_true):
        out.print(f"{GREEN}ACCEPT{RESET}: carried value β_d = {int(beta)} matches W̃_d(rho_d) = {int(beta_true)}.")
        return True
    else:
        out.print(f"{RED}REJECT{RESET}: carried value β_d = {int(beta)} differs from W̃_d(rho_d) = {int(beta_true)}.")
        return False

def _as_int_mod(field: Domain, x: Union[int, ModularInteger],) -> int:
    p = int(field.mod)
    return int(x) % p


def _inv_mod(a: int, p: int) -> Optional[int]:
    """Return a^{-1} mod p, or None if a == 0 mod p."""
    a %= p
    if a == 0:
        return None
    # p is prime in your setting; pow works.
    return pow(a, -1, p)


def dishonest_folding_values(
    *,
    field: Domain,          # GF(p, symmetric=False)
    A: Union[int, ModularInteger],            # add~_i(rho_i, r_x, r_y)
    M: Union[int, ModularInteger],            # mult~_i(rho_i, r_x, r_y)
    beta: Union[int, ModularInteger],         # g_i^*(r_x, r_y)  (the sum-check terminal claim)
    max_tries: int = 50,
) -> Optional[Tuple[ModularInteger, ModularInteger]]:
    """
    Choose (z_x^*, z_y^*) in GF(p) such that:

        beta = A*(z_x^* + z_y^*) + M*z_x^* z_y^*  (mod p).

    Returns:
        (z_x_star, z_y_star) as field elements, or None if impossible / failed.
    """
    p = int(getattr(field, "mod", 0))
    if p <= 1:
        raise ValueError("field must look like GF(p) with .mod.")
    # normalize to the exact sympy GF domain you use elsewhere
    field = field if isinstance(field, FiniteField) else field
    # (If you ever pass a Domain that's not GF, this will still fail later anyway.)

    A_i = _as_int_mod(field, A)
    M_i = _as_int_mod(field, M)
    b_i = _as_int_mod(field, beta)

    # Degenerate: no dependence on zx,zy
    if A_i == 0 and M_i == 0:
        if b_i != 0:
            return None
        zx = random.randrange(p)
        zy = random.randrange(p)
        return field.convert(zx), field.convert(zy)

    # Purely linear: tau = A*(zx+zy)
    if M_i == 0:
        if A_i == 0:
            # already handled above, but keep it safe
            return None
        invA = _inv_mod(A_i, p)
        assert invA is not None
        target_sum = (b_i * invA) % p
        zx = random.randrange(p)
        zy = (target_sum - zx) % p
        return field.convert(zx), field.convert(zy)

    # Purely multiplicative: tau = M*zx*zy
    if A_i == 0:
        invM = _inv_mod(M_i, p)
        assert invM is not None
        if b_i == 0:
            # easiest: pick zx random, set zy = 0
            zx = random.randrange(p)
            return field.convert(zx), field.convert(0)
        # need zx != 0 so we can invert
        for _ in range(max_tries):
            zx = random.randrange(1, p)
            inv_zx = _inv_mod(zx, p)
            assert inv_zx is not None
            zy = (b_i * invM * inv_zx) % p
            return field.convert(zx), field.convert(zy)
        return None

    # General case: tau = A*zx + zy*(A + M*zx)
    # Solve for zy given zx when denom != 0:
    #   zy = (tau - A*zx) / (A + M*zx)
    for _ in range(max_tries):
        zx = random.randrange(p)
        denom = (A_i + M_i * zx) % p

        if denom == 0:
            # Then equation reduces to tau == A*zx (since zy term vanishes)
            if (A_i * zx) % p != b_i:
                continue
            # Any zy works
            zy = random.randrange(p)
            return field.convert(zx), field.convert(zy)

        inv_denom = _inv_mod(denom, p)
        if inv_denom is None:
            continue

        zy = ((b_i - A_i * zx) % p) * inv_denom % p

        # (Optional) sanity check:
        lhs = b_i
        rhs = (A_i * ((zx + zy) % p) + M_i * (zx * zy % p)) % p
        if lhs != rhs:
            continue

        return field.convert(zx), field.convert(zy)

    return None


def _sorted_syms_with_prefix(gens: Sequence[Symbol], prefix: str) -> List[Symbol]:
    """
    Extract symbols like z_0, z_1, ... (or x_*, y_*) from a gens list and sort by index.
    """
    syms = [s for s in gens if str(s).startswith(prefix)]
    def key(s: Symbol) -> int:
        name = str(s)
        # "x_12" -> 12
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9
    return sorted(syms, key=key)


@dataclass
class LayerTransitionResult:
    ok: bool
    i: int
    rho_i: Tuple[Union[int, ModularInteger], ...]
    beta_i: Union[int, ModularInteger]

    # sum-check transcript pieces
    r_x: Tuple[Union[int, ModularInteger], ...]
    r_y: Tuple[Union[int, ModularInteger], ...]

    # wiring predicate scalars at (rho_i, r_x, r_y)
    A: Union[int, ModularInteger]
    M: Union[int, ModularInteger]

    # bridging value from sum-check transcript
    gamma: Union[int, ModularInteger]

    # q-polynomials (optional but you want them)
    q: Optional[Poly] = None
    q_star: Optional[Poly] = None

    # endpoints: true + claimed
    z_x_true: Union[int, ModularInteger] = 0
    z_y_true: Union[int, ModularInteger] = 0
    zx_star: Union[int, ModularInteger] = 0
    zy_star: Union[int, ModularInteger] = 0

    # verifier folding challenge
    tau: Optional[Union[int, ModularInteger]] = None

    # next claim
    rho_next: Tuple[Union[int, ModularInteger], ...] = ()
    beta_next: Union[int, ModularInteger] = 0


def normalize_to_x_vars(poly: Poly, n: int) -> Poly:
    orig = list(poly.gens)
    if len(orig) != n:
        raise ValueError(f"Expected {n} gens, got {len(orig)}: {orig}")
    x_syms = list(symbols(f"x_:{n}"))
    repl = dict(zip(orig, x_syms))
    expr = poly.as_expr().subs(repl)
    return Poly(expr, *x_syms, domain=poly.domain)

def gkr_layer_transition(
    ac: ArithmeticCircuit,
    *,
    i: int,
    rho_i: Tuple[Union[int, ModularInteger], ...],
    beta_i: Union[int, ModularInteger],
    honest_prover: bool = True,
    # if you want to demo cheating later:
    dishonest_folding: bool = False,
    out: TerminalOutput = default_out,
) -> LayerTransitionResult:
    """
    One GKR layer transition: from claim
        beta_i = W~_i(rho_i)
    to a new claim
        beta_{i+1} = W~_{i+1}(rho_{i+1})
    using Thaler's identity + sum-check + folding.
    """

    F = ac.field  # GF(p, symmetric=False)
    beta_i_F = F.convert(int(beta_i))

    # ---- determine arities s_i and s_{i+1} (bit-lengths) from the circuit dictionaries ----
    # W_dict[i] keys are tuples of bits; their length is s_i.
    s_i = len(next(iter(ac.W_dict[i].keys())))
    s_next = len(next(iter(ac.W_dict[i + 1].keys())))

    if len(rho_i) != s_i:
        raise ValueError(f"rho_i must have length s_i={s_i}, got {len(rho_i)}")

    # ---- fetch polynomials for this layer ----
    W_next = ac.tilde_W[i + 1]         # in x_0..x_{s_next-1}
    add_i = ac.tilde_add[i]            # in z_*, x_*, y_*
    mult_i = ac.tilde_mult[i]          # in z_*, x_*, y_*

    # ---- fetch polynomials for this layer ----
    W_next = ac.tilde_W[i + 1]         # in x_0..x_{s_next-1} (under-the-hood names)
    add_i_raw = ac.tilde_add[i]        # in gens of length s_i + 2*s_next (under-the-hood)
    mult_i_raw = ac.tilde_mult[i]      # same

    # ---- rename wiring predicate polynomials into the canonical (z_*, x_*, y_*) ring ----
    add_i = replace_symbols_in_polynomial(add_i_raw, v=s_i, w=s_next)     # gens: z_0.., x_0.., y_0..
    mult_i = replace_symbols_in_polynomial(mult_i_raw, v=s_i, w=s_next)

    z_syms = list(symbols(f"z_:{s_i}"))
 
    # We'll run sum-check in gens (x_0.., y_0..)
    x_syms = list(symbols(f"x_:{s_next}"))
    y_syms = list(symbols(f"y_:{s_next}"))

    # Substitute z := rho_i into add~, mult~ to get polynomials in x,y only
    z_subs = {z_syms[j]: F.convert(int(rho_i[j])) for j in range(s_i)}

    add_xy_expr = add_i.as_expr().subs(z_subs)
    mult_xy_expr = mult_i.as_expr().subs(z_subs)

    add_xy = Poly(add_xy_expr, *(x_syms + y_syms), domain=F)
    mult_xy = Poly(mult_xy_expr, *(x_syms + y_syms), domain=F)

    # Build W_next(x) and W_next(y)
    W_next = normalize_to_x_vars(ac.tilde_W[i + 1], s_next)  # gens: x_0..x_{s_next-1}
    Wx_expr = W_next.as_expr().subs({W_next.gens[j]: x_syms[j] for j in range(s_next)})
    Wy_expr = W_next.as_expr().subs({W_next.gens[j]: y_syms[j] for j in range(s_next)})

    # Thaler identity integrand:
    # g_i(x,y) = add~(rho,x,y)*(W(x)+W(y)) + mult~(rho,x,y)*(W(x)*W(y))
    g_expr = add_xy.as_expr() * (Wx_expr + Wy_expr) + mult_xy.as_expr() * (Wx_expr * Wy_expr)
    g_poly = Poly(g_expr, *(x_syms + y_syms), domain=F)

    # ---- run sum-check on g_poly, claiming sum_{x,y in {0,1}^{s_next}} g_poly(x,y) == tau_i ----
    print_header(f"GKR LAYER {i} -> {i+1}: SUM-CHECK", level=1, out=out)
    out.print(f"\nWe reduce β_{i} = W̃_{i}(ρ_{i}) to a sum over Boolean x,y via Thaler's identity.")
    out.print(f"\nSum-check polynomial has {2*s_next} variables: x_0..x_{s_next-1}, y_0..y_{s_next-1}.")
    out.print(f"\nClaimed sum H* := β_{i} = {int(beta_i_F)}.")

    r: List[Union[int, ModularInteger]] = []
    gamma = sum_check_recursion(
        g_init=g_poly,
        g=g_poly,
        H_star=beta_i_F,   # claimed sum is still β_i
        r=r,
        dishonest=[False],
        user_is_verifier=False,
        user_is_prover=False,
        show_steps=True,
        out=out,
        return_final_value=True,
        show_final_check=False,
    )

    if gamma is False:
        return LayerTransitionResult(
            ok=False,
            i=i,
            rho_i=rho_i,
            beta_i=beta_i,
        )

    # gamma is now g*_{m-1}(r_{m-1}) as a field element
    gamma_F = F.convert(int(gamma))

    # r now has length 2*s_next; split into r_x, r_y
    r_x = tuple(r[:s_next])
    r_y = tuple(r[s_next:])

    # Evaluate A, M at (rho_i, r_x, r_y)
    A_val = int(F(add_xy.eval({x_syms[j]: F.convert(int(r_x[j])) for j in range(s_next)} |
                              {y_syms[j]: F.convert(int(r_y[j])) for j in range(s_next)})))
    M_val = int(F(mult_xy.eval({x_syms[j]: F.convert(int(r_x[j])) for j in range(s_next)} |
                               {y_syms[j]: F.convert(int(r_y[j])) for j in range(s_next)})))

    t = symbols("t")

    # ---- build the line ℓ(t) = (1 - t) r_x + t r_y ----
    # (Convention: ℓ(0)=r_x and ℓ(1)=r_y.)
    line_subs = {
        W_next.gens[j]: ((1 - t) * F.convert(int(r_x[j])) + t* F.convert(int(r_y[j])))
        for j in range(s_next)
    }

    # ---- honest q_{i+1}(t) = W̃_{i+1}(ℓ(t)) ----
    q = Poly(W_next.as_expr().subs(line_subs), t, domain=F)

    # endpoints implied by q (note: q_{i+1}(0)=W̃_{i+1}((r_x), q_{i+1}(1)=W̃_{i+1}((r_y))
    q_at_0 = F.convert(q.eval({t: F.convert(0)}))  # = W̃_{i+1}(r_x)
    q_at_1 = F.convert(q.eval({t: F.convert(1)}))  # = W̃_{i+1}(r_y)

    # True values (omniscient / simulator)
    z_x = int(q_at_0)
    z_y = int(q_at_1)

    # Honest (or dishonest) folding values z_x = W̃_{i+1}(r_x), z_y = W̃_{i+1}(r_y)
    if honest_prover and not dishonest_folding:
        # Prover sends the *true* polynomial and consistent endpoints.
        q_star = q
        zx_star = z_x   # claimed W̃_{i+1}(r_x)
        zy_star = z_y   # claimed W̃_{i+1}(r_y)
    else:
        # Prover chooses dishonest endpoints (helper can do this),
        # then sets the cheapest consistent q^*(t).
        zx_star, zy_star = dishonest_folding_values(
            field=F,
            A=F.convert(A_val),
            M=F.convert(M_val),
            tau=gamma_F,
            out=out,
        )
        zx_star = int(F.convert(int(zx_star)))
        zy_star = int(F.convert(int(zy_star)))

        # Dishonest folding polynomial: q^*(t) = (1 - t) z*_x + t z*_y
        q_star = Poly((1 - t) * F.convert(zx_star) + t * F.convert(zy_star), t, domain=F)

    deg_bound = s_next

    if q_star.degree() > deg_bound:
        out.print(f"\nVerifier {RED}REJECTS{RESET}: deg(q*_{i+1}) = {q_star.degree()} > {deg_bound}.")
        return LayerTransitionResult(
            ok=False, i=i, rho_i=rho_i, beta_i=beta_i,
            r_x=r_x, r_y=r_y, A=A_val, M=M_val, z_x=zx_star, z_y=zy_star, rho_next=(), beta_next=0
        )

    q0 = F.convert(q_star.eval({t: F.convert(0)}))
    q1 = F.convert(q_star.eval({t: F.convert(1)}))

    # Because ℓ(0)=r_x and ℓ(1)=r_y, the endpoint checks are:
    if int(q0) != int(F.convert(zx_star)) or int(q1) != int(F.convert(zy_star)):
        out.print(f"\nVerifier {RED}REJECTS{RESET}: q*_{i+1}(0/1) do not match z*_x/z*_y.")
        return LayerTransitionResult(
            ok=False, 
            i=i, 
            rho_i=rho_i, 
            beta_i=beta_i,
            gamma=lhs,
            q=q,
            q_star=q_star,
            z_x_true=z_x,
            z_y_true=z_y,
            zx_star=zx_star, 
            zy_star=zy_star, 
            tau=None,
            r_x=r_x, 
            r_y=r_y, 
            A=A_val, 
            M=M_val, 
            rho_next=(), 
            beta_next=0
        )

    # Verifier checks the folding equation:
    lhs = int(gamma_F)
    rhs = int(
        F.convert(A_val) * (F.convert(zx_star) + F.convert(zy_star))
        + F.convert(M_val) * (F.convert(zx_star) * F.convert(zy_star))
    )

    out.print(f"\nBRIDGING CHECK at layer {i}:")
    out.print(f"γ := g*_{2*s_next - 1}(r_{2*s_next - 1}) = {lhs}")
    out.print(f"A := add̃_{i}(ρ_i,r_x,r_y) = {A_val}")
    out.print(f"M := mult̃_{i}(ρ_i,r_x,r_y) = {M_val}")
    out.print(f"z*_x := (claimed) W̃_{i+1}(r_x) = {zx_star}")
    out.print(f"z*_y := (claimed) W̃_{i+1}(r_y) = {zy_star}")
    out.print(f"\nCheck: γ ?= A(z*_x+z*_y) + M(z*_x z*_y)")
    out.print(f"{lhs} ?= {rhs}")

    if lhs != rhs:
        out.print(f"\nVerifier {RED}REJECTS{RESET} (bridging equation failed).")
        return LayerTransitionResult(
            ok=False, 
            i=i, 
            rho_i=rho_i, 
            beta_i=beta_i,
            gamma=lhs,
            q=q,
            q_star=q_star,
            z_x_true=z_x,
            z_y_true=z_y,
            zx_star=zx_star, 
            zy_star=zy_star, 
            tau=None,
            r_x=r_x, 
            r_y=r_y, 
            A=A_val, 
            M=M_val, 
            rho_next=(), 
            beta_next=0
        )

    out.print(f"\nBridging equation {GREEN}PASSED{RESET}.")

    # Now fold two evaluations into one via a random scalar tau
    tau = random_field_element(field=F, user_input=False, out=out)
    tau_F = F.convert(int(tau))

    rho_next = tuple(
        int(F((1 - tau_F) * F.convert(int(r_x[j])) + tau_F * F.convert(int(r_y[j]))))
        for j in range(s_next)
    )

    beta_next = int(F.convert(q_star.eval({t: tau_F})))

    out.print(f"\nVerifier samples τ uniformly at random from {F}; here τ = {int(tau_F)}.")
    out.print(f"\nDefines ρ_{i+1} := (1-τ) r_x + τ r_y  ∈ F^{s_next}.")
    out.print(f"New claim: β_{i+1} := q*_{i+1}(τ) = {beta_next}.")
    
    return LayerTransitionResult(
        ok=True, 
        i=i, 
        rho_i=rho_i, 
        beta_i=beta_i,
        gamma=lhs,
        q=q,
        q_star=q_star,
        z_x_true=z_x,
        z_y_true=z_y,
        tau=int(tau_F),
        r_x=r_x, 
        r_y=r_y, 
        A=A_val, 
        M=M_val, 
        zx_star=zx_star, 
        zy_star=zy_star, 
        rho_next=rho_next, 
        beta_next=beta_next
    )