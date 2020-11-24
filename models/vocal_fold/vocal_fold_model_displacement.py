# -*- coding: utf-8 -*-
from typing import List


def vdp_coupled(t: float, Z: List[float], alpha: float, beta: float, delta: float) -> List[float]:
    """ Physical model of the displacement of vocal folds.
    The model is in the explicit form of a pair of coupled van der Pol oscillators:
        dZ = f(Z)
    which include two second order, nonlinear, constant coefficients, inhomogeneous ODEs.

    Args:
        t: float
            Time.
        Z: List[float]
            State variables [u1(t), u2(t), v1(t), v2(t)], u c.r.t right, v c.r.t. left.
        alpha: float
            Glottal pressure coupling parameter.
        beta: float
            Mass, damping, stiffness parameter.
        delta: float
            Asymmetry parameter.

    Returns:
        dZ: List[float]
            Drivatives of state variables [du1, du2, dv1, dv2].
    """
    du1 = Z[1]

    dv1 = Z[3]

    du2 = -beta * (1 + Z[0] ** 2) * Z[1] - (1 - delta / 2) * Z[0] + alpha * (Z[1] + Z[3])

    dv2 = -beta * (1 + Z[2] ** 2) * Z[3] - (1 + delta / 2) * Z[2] + alpha * (Z[1] + Z[3])

    dZ = [du1, du2, dv1, dv2]
    return dZ


def vdp_jacobian(
    t: float, Z: List[float], alpha: float, beta: float, delta: float
) -> List[List[float]]:
    """ Jacobian of the above system of the form:
            J[i, j] = df[i] / dZ[j]
    """
    J = [
        [0, 1, 0, 0],
        [-2 * beta * Z[1] * Z[0] - (1 - delta / 2), -beta * (1 + Z[0] ** 2) + alpha, 0, alpha],
        [0, 0, 0, 1],
        [0, alpha, -2 * beta * Z[3] * Z[2] - (1 + delta / 2), -beta * (1 + Z[2] ** 2) + alpha],
    ]

    return J
