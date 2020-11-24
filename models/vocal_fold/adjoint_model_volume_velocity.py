# -*- coding: utf-8 -*-

from typing import List

import numpy as np


def adjoint_model(
    t: float,
    L: List[float],
    dL: List[float],
    alpha: float,
    beta: float,
    delta: float,
    Ru_Sm_duR: float,
) -> np.ndarray:
    """ Adjoint model for vocal fold velocity model.
    Used to solve the derivatives of volume velocity to model parameters.

    Args:
        t: float
            Time.
        L: List[float]
            State variables [l1(t), l2(t)].
        dL: List[float]
            Derivatives of state variables [dl1(t), dl2(t)]/
        alpha: float
            Glottal pressure coupling parameter.
        beta: float
            Mass, damping, stiffness parameter.
        delta: float
            Asymmetry parameter.
        Ru_Sm_duR: float
            Extra terms required for computation.

    Returns:
        res: np.ndarray[float]
            Residual vector.
    """
    res_1 = dL[1] + (delta + 1) * L[0] + 2 * Ru_Sm_duR

    res_2 = (beta - 2 * alpha) * L[0]

    return np.array([res_1, res_2])
