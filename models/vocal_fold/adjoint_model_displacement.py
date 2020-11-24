# -*- coding: utf-8 -*-
from typing import Callable, List

import numpy as np


def adjoint_model(
    alpha: float,
    beta: float,
    delta: float,
    X: List[List[float]],
    dX: List[List[float]],
    R: List[float],
    fs: int,
    t0: float,
    tf: float,
):
    """ Adjoint model for the 1-d vocal fold displacement model.
    Used to solve the derivatives of right/left vocal fold displacements w.r.t. 
        model parameters (alpha, beta, delta).

    Args:
        alpha: float
            Glottal pressure coupling parameter.
        beta: float
            Mass, damping, stiffness parameter.
        delta: float
            Asymmetry parameter.
        X: List[List[float]]
            Vocal fold displacements [x_r, x_l].
        dX: List[List[fliat]]
            Vocal fold velocity [dx_r, dx_l].
        R: List[float]
            Term c.r.t. the difference between predicted and actual volume velocity flows.
        fs: int
            Sample rate.
        t0: float
            Start time.
        tf: float
            Stop time.

    Returns:
        residual: Callable[[float, List[float], List[float]], np.ndarray]
            Adjoint model.
        jac: Callable[[float, float, List[float], List[float]], np.ndarray]
            Jacobian of the adjoint model.
    """

    def residual(t: float, M: List[float], dM: List[float]) -> np.ndarray:
        """ Defines the adjoint model, which should be in the implicit form:
                0 <-- res = F(t, M, dM)

        Args:
            t: float
                Time.
            M: List[float]
                State variables [L, dL, E, dE].
            dM: List[float]
                Derivatives of state variables [dL, ddL, dE, ddE].

        Returns:
            res: np.ndarray[float], shape (len(M),)
                Residual vector.
        """
        # Convert t to [0, T]
        t = t - t0
        # Convert t(s) to idx(#sample)
        idx = int(round(t * fs) - 1)
        if idx < 0:
            idx = 0
        # print(f't: {t:.4f}    adjoint idx: {idx:d}')

        x = X[idx]
        dx = dX[idx]
        r = R[idx]

        res_1 = dM[1] + (2 * beta * x[0] * dx[0] + 1 - 0.5 * delta) * M[0] + r

        res_2 = beta * M[0] * (1 + x[0] ** 2) - alpha * (M[0] + M[2])

        res_3 = dM[3] + (2 * beta * x[1] * dx[1] + 1 + 0.5 * delta) * M[2] + r

        res_4 = beta * M[2] * (1 + x[1] ** 2) - alpha * (M[0] + M[2])

        res = np.array([res_1, res_2, res_3, res_4])

        return res

    def jac(c: float, t: float, M: List[float], Md: List[float]) -> np.ndarray:
        """ Defines the Jacobian of the adjoint model, which should be in the form:
                J = dF/dM + c*dF/d(dM)

        Args:
            c: float
                Constant.
            t: float
                Time.
            M: List[float]
                State variables [L, dL, E, dE].
            dM: List[float]
                Derivative of state variables [dL, ddL, dE, ddE].

        Returns:
            jacobian: np.ndarray[float], shape (len(M), len(M))
                Jacobian matrix.
        """
        # Convert t to [0, T]
        T = tf - t0
        t = (t - t0) / (tf - t0) * T
        # Convert t(s) to idx(#sample)
        idx = int(round(t * fs) - 1)
        if idx < 0:
            idx = 0

        x = X[idx]
        dx = dX[idx]

        jacobian = np.zeros((len(M), len(M)))

        # jacobian[0, 0] = 2 * beta * x[0] * dx[0] + 1 - 0.5 * delta
        # jacobian[0, 1] = c
        # jacobian[1, 2] = 2 * beta * x[1] * dx[1] + 1 + 0.5 * delta
        # jacobian[1, 3] = c
        # jacobian[2, 0] = beta * (1 + x[0] ** 2) - alpha
        # jacobian[2, 2] = -alpha
        # jacobian[3, 0] = -alpha
        # jacobian[3, 2] = beta * (1 + x[1] ** 2) - alpha

        jacobian[0, 0] = 2 * beta * x[0] * dx[0] + 1 - 0.5 * delta
        jacobian[0, 1] = c

        jacobian[1, 0] = beta * (1 + x[0] ** 2) - alpha
        jacobian[1, 2] = -alpha

        jacobian[2, 2] = 2 * beta * x[1] * dx[1] + 1 + 0.5 * delta
        jacobian[2, 3] = c

        jacobian[3, 0] = -alpha
        jacobian[3, 2] = beta * (1 + x[1] ** 2) - alpha

        return jacobian

    return residual, jac
