# -*- coding: utf-8 -*-
from typing import Callable, List, Optional

import numpy as np
from assimulo.problem import Implicit_Problem
from matplotlib import pyplot as plt


def dae_solver(
    residual: Callable,
    y0: List[float],
    yd0: List[float],
    t0: float,
    tfinal: float = 10.0,
    backward: bool = False,
    ncp: int = 500,
    solver: str = "IDA",
    algvar: Optional[List[bool]] = None,
    suppress_alg: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    usejac: bool = False,
    jac: Optional[Callable] = None,
    usesens: bool = False,
    sensmethod: str = "STAGGERED",
    p0: Optional[List[float]] = None,
    pbar: Optional[List[float]] = None,
    suppress_sens: bool = False,
    display_progress: bool = True,
    report_continuously: bool = False,
    verbosity: int = 30,
    name: str = "DAE",
) -> List[float]:
    """ DAE solver.

    Args:
        residual: Callable
            Implicit DAE model.
        y0: List[float]
            Initial model state.
        yd0: List[float]
            Initial model state derivatives.
        t0: float
            Initial simulation time.
        tfinal: float
            Final simulation time.
        backward: bool
            Specifies if the simulation is done in reverse time.
        ncp: int
            Number of communication points (number of returned points).
        solver: str
            Solver name.
        algvar: List[bool]
            Defines which variables are differential and which are algebraic.
            The value True(1.0) indicates a differential variable and the value False(0.0) indicates an algebraic variable.
        suppress_alg: bool
            Indicates that the error-tests are suppressed on algebraic variables.
        atol: float
            Absolute tolerance.
        rtol: float
            Relative tolerance.
        usejac: bool
            Whether to use the user defined jacobian.
        jac: Callable
            Model jacobian.
        usesens: bool
            Aactivates or deactivates the sensitivity calculations.
        sensmethod: str
            Specifies the sensitivity solution method.
            Can be either ‘SIMULTANEOUS’ or ‘STAGGERED’. Default is 'STAGGERED'.
        p0: List[float]
            Parameters for which sensitivites are to be calculated.
        pbar: List[float]
            An array of positive floats equal to the number of parameters. Default absolute values of the parameters.
            Specifies the order of magnitude for the parameters. Useful if IDAS is to estimate tolerances for the sensitivity solution vectors.    
        suppress_sens: bool
            Indicates that the error-tests are suppressed on the sensitivity variables.
        display_progress: bool
            Actives output during the integration in terms of that the current integration is periodically printed to the stdout.
            Report_continuously needs to be activated.
        report_continuously: bool
            Specifies if the solver should report the solution continuously after steps.    
        verbosity: int
            Determines the level of the output.
            QUIET = 50 WHISPER = 40 NORMAL = 30 LOUD = 20 SCREAM = 10.
        name: str
            Model name.

    Returns:
        sol: List[float]
            Solution [time, model states].
    """
    if usesens is True:  # parameter sensitivity
        model = Implicit_Problem(residual, y0, yd0, t0, p0=p0)
    else:
        model = Implicit_Problem(residual, y0, yd0, t0)

    model.name = name

    if algvar is not None:  # differential or algebraic variables
        model.algvar = algvar

    if usejac is True:  # jacobian
        model.jac = jac

    if solver == "IDA":  # solver
        from assimulo.solvers import IDA

        sim = IDA(model)

    sim.backward = backward  # backward in time
    sim.suppress_alg = suppress_alg
    sim.atol = atol
    sim.rtol = rtol
    sim.display_progress = display_progress
    sim.report_continuously = report_continuously
    sim.verbosity = verbosity

    if usesens is True:  # sensitivity
        sim.sensmethod = sensmethod
        sim.pbar = np.abs(p0)
        sim.suppress_sens = suppress_sens

    # Simulation
    # t, y, yd = sim.simulate(tfinal, ncp=(ncp - 1))
    ncp_list = np.linspace(t0, tfinal, num=ncp, endpoint=True)
    t, y, yd = sim.simulate(tfinal, ncp=0, ncp_list=ncp_list)

    # Plot
    # plt.figure()
    # plt.subplot(221)
    # plt.plot(t, y[:, 0], 'b.-')
    # plt.legend([r'$\lambda$'])
    # plt.subplot(222)
    # plt.plot(t, y[:, 1], 'r.-')
    # plt.legend([r'$\dot{\lambda}$'])
    # plt.subplot(223)
    # plt.plot(t, y[:, 2], 'k.-')
    # plt.legend([r'$\eta$'])
    # plt.subplot(224)
    # plt.plot(t, y[:, 3], 'm.-')
    # plt.legend([r'$\dot{\eta}$'])

    # plt.figure()
    # plt.subplot(221)
    # plt.plot(t, yd[:, 0], 'b.-')
    # plt.legend([r'$\dot{\lambda}$'])
    # plt.subplot(222)
    # plt.plot(t, yd[:, 1], 'r.-')
    # plt.legend([r'$\ddot{\lambda}$'])
    # plt.subplot(223)
    # plt.plot(t, yd[:, 2], 'k.-')
    # plt.legend([r'$\dot{\eta}$'])
    # plt.subplot(224)
    # plt.plot(t, yd[:, 3], 'm.-')
    # plt.legend([r'$\ddot{\eta}$'])

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(y[:, 0], y[:, 1])
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$\dot{\lambda}$')
    # plt.subplot(122)
    # plt.plot(y[:, 2], y[:, 3])
    # plt.xlabel(r'$\eta$')
    # plt.ylabel(r'$\dot{\eta}$')

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(yd[:, 0], yd[:, 1])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\ddot{\lambda}$')
    # plt.subplot(122)
    # plt.plot(yd[:, 2], yd[:, 3])
    # plt.xlabel(r'$\dot{\eta}$')
    # plt.ylabel(r'$\ddot{\eta}$')

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(y[:, 0], y[:, 2])
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$\eta$')
    # plt.subplot(122)
    # plt.plot(y[:, 1], y[:, 3])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\dot{\eta}$')

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(yd[:, 0], yd[:, 2])
    # plt.xlabel(r'$\dot{\lambda}$')
    # plt.ylabel(r'$\dot{\eta}$')
    # plt.subplot(122)
    # plt.plot(yd[:, 1], yd[:, 3])
    # plt.xlabel(r'$\ddot{\lambda}$')
    # plt.ylabel(r'$\ddot{\eta}$')

    # plt.show()

    sol = [t, y, yd]
    return sol
