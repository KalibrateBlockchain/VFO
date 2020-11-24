# -*- coding: utf-8 -*-
from typing import List


def vocal_fold_model(
    U: List[float], t: float, alpha: float, beta: float, delta: float
) -> List[float]:
    """ Model of volume velocity flow through vocal folds.
    
    Args:
        U: List[float]
            State variables [u1(t), u2(t)].
        t: float
            Time.
        alpha: float
            Glottal pressure coupling parameter.
        beta: float
            Mass, damping, stiffness parameter.
        delta: float
            Asymmetry parameter.

    Returns:
        dU: List[float]
            Drivatives of state variables [du1, du2].
    """
    du1 = U[1]

    du2 = -(beta - 2 * alpha) * U[1] - (delta + 1) * U[0]

    dU = [du1, du2]

    return dU


def vdp_jacobian(
    U: List[float], t: float, alpha: float, beta: float, delta: float
) -> List[List[float]]:
    """ Jacobian of the above system:
            J[i, j] = d(dU[i]) / d(U[j])
    """
    J = [[0, 1], [-(delta + 1), -(beta - 2 * alpha)]]

    return J
