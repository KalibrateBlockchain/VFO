# -*- coding: utf-8 -*-
from typing import Callable, List

import numpy as np
from scipy.integrate import ode


def ode_solver(
    model: Callable,
    model_jacobian: Callable,
    model_params: List[float],
    init_state: List[float],
    init_t: float,
    solver: str = "lsoda",
    ixpr: int = 1,
    dt: float = 0.1,
    tmax: float = 1000,
) -> np.ndarray:
    """ ODE solver.

    Args:
        model: Callable
            ODE model dy = f(t, y).
        model_jacobian: Callable
            Jacobian of ODE model.
        model_params: List[float]
            Model parameters.
        init_state: List[float]
            Initial model state.
        init_t: float
            Initial simulation time.
        solver: str
            Solver name. Options: vode, dopri5, dop853, lsoda; depends on stiffness and precision.
        ixpr: int
            Whether to generate extra printing at method switches.
        dt: float
            Time step increment.
        tmax: float
            Maximum simulation time.

    Returns:
        sol: np.ndarray[float]
            Solution [time, model states].
    """
    sol = []

    r = ode(model, model_jacobian)

    r.set_f_params(*model_params)
    r.set_jac_params(*model_params)
    r.set_initial_value(init_state, init_t)
    r.set_integrator(solver, with_jacobian=True, ixpr=ixpr)

    while r.successful() and r.t < tmax:
        r.integrate(r.t + dt)
        sol.append([r.t, *list(r.y)])

    return np.array(sol)  # (t, [p, dp]) tangent bundle
