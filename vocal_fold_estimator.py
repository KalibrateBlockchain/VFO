import numpy as np
from scipy.io import wavfile

from models.vocal_fold.adjoint_model_displacement import adjoint_model
from models.vocal_fold.vocal_fold_model_displacement import (
    vdp_coupled,
    vdp_jacobian,
)
from solvers.ode_solvers.dae_solver import dae_solver
from solvers.ode_solvers.ode_solver import ode_solver
from solvers.optimization import optim_adapt_step, optim_grad_step
from math import floor, ceil
import logging


np.random.seed(123)

#CISCO def vocal_fold_estimator(wav_file_path,glottal_flow, logger, t_patience = 500, section = 1):
def vocal_fold_estimator(wav_file_path, wav_samples,sample_rate,glottal_flow, logger, t_patience = 5, section = 1):
    """
    Inputs: wav_file_path: Path to read from actual wavfile
            glottal_flow: numpy array of glottal flow from IAIF

    returns: dictionary best_results:
    ["iteration", "R", "Rk", "alpha", "beta", "delta", "sol", "u0"]
    """
    
    # Set constants
    M = 0.5  # mass, g/cm^2
    B = 100  # damping, dyne s/cm^3
    d = 1.75  # length of vocal folds, cm
    x0 = 0.1  # half glottal width at rest position, cm
    tau = 1e-3  # time delay for surface wave to travel half glottal height, ms
    c = 5000  # air particle velocity, cm/s
    eta = 1.0  # nonlinear factor for energy dissipation at large amplitude

    """CISCO
    sample_rate, wav_samples = wavfile.read(wav_file_path)
    if section == 1:
        wav_samples = wav_samples[floor(len(wav_samples)/2): ceil(len(wav_samples)/2 + sample_rate)]
    """
    


    # NOTE: If you want to plot glottal flow and original waveform together
    # fig = plt.figure()
    # plt.plot(np.linspace(0, len(wav_samples) / sample_rate, len(wav_samples)), wav_samples)
    # plt.plot(np.linspace(0, len(glottal_flow) / sample_rate, len(glottal_flow)), glottal_flow)
    # plt.legend(["speech sample", "glottal flow"])
    # plt.show()

    # Set model initial conditions
    delta = np.random.random()  # asymmetry parameter
    alpha = 0.6 * delta  # if > 0.5 delta, stable-like oscillator
    beta = 0.2

    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
    num_tsteps = len(wav_samples)  # total number of time steps
    T = len(wav_samples) / float(sample_rate)  # total time, s
    logger.info(
        f"Initial parameters: alpha = {alpha:.4f}   beta = {beta:.4f}   delta = {delta:.4f}"
    )
    logger.debug("-" * 110)

    # Optimize
    best_results: Dict[str, List[float]] = {  # store best results over iterations
        "iteration": [],  # optimize iter
        "R": [],  # estimation residual @ k
        "Rk": [],  # estimation residual w.r.t L2 norm @ k
        "alpha": [],
        "beta": [],
        "delta": [],
        "sol": [],  # model ouputs
        "u0": [],  # estimated glottal flow
    }
    iteration = 0
    Rk = 1e16
    Rk_best = 1e16
    patience = 0  # number of patient iterations of no improvement before stopping optimization
    if_adjust = 0

    while patience < t_patience: # this was 400 default
        # Solve vocal fold displacement model
        # logger.info("Solving vocal fold displacement model")
        K = B ** 2 / (beta ** 2 * M)
        Ps = (alpha * x0 * np.sqrt(M * K)) / tau
        #time_scaling = np.sqrt(K / float(M))  # t -> s
        time_scaling = np.sqrt(K / float(M))*100  # t -> s

        x_scaling = np.sqrt(eta)
        # logger.debug(
        #     f"stiffness K = {K:.4f} dyne/cm^3    subglottal Ps = {Ps:.4f} dyne/cm^2    "
        #     f"time_scaling = {time_scaling:.4f}"
        # )

        vdp_params = [alpha, beta, delta]
        sol = ode_solver(
            vdp_coupled,
            vdp_jacobian,
            vdp_params,
            vdp_init_state,
            (time_scaling * vdp_init_t),
            solver="lsoda",
            ixpr=0,
            dt=(time_scaling / float(sample_rate)),  # dt -> ds
            tmax=(time_scaling * T),
        )

        if len(sol) > len(wav_samples):
            sol = sol[:-1]
        assert len(sol) == len(
            wav_samples
        ), f"Inconsistent length: ODE sol ({len(sol):d}) / wav samples ({len(wav_samples):d})"

        # Calculate glottal flow
        try:
            assert sol.size > 0
            X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
            dX = sol[:, [2, 4]]  # cm/s
            u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
            u0 = u0 / np.linalg.norm(u0) * np.linalg.norm(glottal_flow)  # normalize
        except AssertionError as e:
            logger.error(e)
            logger.warning("Skip")
            break

        # Estimation residual
        R = u0 - glottal_flow

        # NOTE: If you want to plot glottal flow by IAIF vs estimated glottal flow
        # plt.figure()
        # plt.plot(sol[:, 0], glottal_flow, "k.-")
        # plt.plot(sol[:, 0], u0, "b.-")
        # plt.plot(sol[:, 0], R, "r.-")
        # plt.xlabel("t")
        # plt.legend(["glottal flow", "estimated glottal flow", "residual"])
        # plt.show()

        # Solve adjoint model
        # logger.info("Solving adjoint model")

        residual, jac = adjoint_model(alpha, beta, delta, X, dX, R, sample_rate, 0, T)
        M_T = [0.0, 0.0, 0.0, 0.0]  # initial states of adjoint model at T
        dM_T = [0.0, -R[-1], 0.0, -R[-1]]  # initial ddL = ddE = -R(T)
        try:
            adjoint_sol = dae_solver(
                residual,
                M_T,
                dM_T,
                T,
                tfinal=0,  # simulate (tfinal-->t0)s backward
                backward=True,
                ncp=len(wav_samples),
                solver="IDA",
                algvar=[0, 1, 0, 1],
                suppress_alg=True,
                atol=1e-6,
                rtol=1e-6,
                usejac=True,
                jac=jac,
                usesens=False,
                display_progress=True,
                report_continuously=False,  # NOTE: report_continuously should be False
                verbosity=50,
            )
        except Exception as e:
            logger.error(f"Exception: {e}")
            logger.warning("Skip")
            break

        # Compute adjoint lagrange multipliers
        L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
        E = adjoint_sol[1][:, 2][::-1]
        assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"
        L = L / np.linalg.norm(L)  # normalize
        E = E / np.linalg.norm(E)

        # Update parameters
        # logger.info("Updating parameters")

        # Record parameters @ current step
        alpha_k = alpha
        beta_k = beta
        delta_k = delta
        Rk = np.sqrt(np.sum(R ** 2))
        print(
            f"[{patience:d}:{iteration:d}] L2 Residual = {Rk:.4f} | alpha = {alpha_k:.4f}   "
            f"beta = {beta_k:.4f}   delta = {delta_k:.4f}"
        )
        
        if Rk < Rk_best:  # has improvement
            # Record best
            iteration_best = iteration
            R_best = R
            Rk_best = Rk
            alpha_best = alpha_k
            beta_best = beta_k
            delta_best = delta_k
            sol_best = sol
            u0_best = u0
            pv_best = np.array([alpha_best, beta_best, delta_best])  # param vector

            # Compute gradients
            d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
            d_beta = np.sum(
                L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0]
                + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1]
            )
            d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vector
            dpv = dpv / np.linalg.norm(dpv)  # normalize
            d_alpha, d_beta, d_delta = dpv

            # Update
            alpha, beta, delta = optim_grad_step(
                alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.1
            )
            # alpha, beta, delta = optim_adapt_step(
            # alpha, beta, delta, d_alpha, d_beta, d_delta, default_step=0.1
            # )
            iteration += 1
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] IMPROV: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )
        else:  # no improvement
            patience = patience + 1

            # Compute conjugate gradients
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vector
            dpv = dpv / np.linalg.norm(dpv)  # normalize
            ov = np.random.randn(len(dpv))  # orthogonal vector
            ov = ov - (np.dot(ov, dpv) / np.dot(dpv, dpv)) * dpv  # orthogonalize
            ov = ov / np.linalg.norm(ov)  # normalize
            d_alpha, d_beta, d_delta = ov

            # Reverse previous update & update in conjugate direction
            alpha, beta, delta = optim_grad_step(
                alpha_best, beta_best, delta_best, d_alpha, d_beta, d_delta, stepsize=0.1,
            )
            # alpha, beta, delta = optim_adapt_step(
            #     alpha_best,
            #     beta_best,
            #     delta_best,
            #     d_alpha,
            #     d_beta,
            #     d_delta,
            #     default_step=0.1,
            # )
            iteration += 1
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] NO IMPROV: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )

        while (alpha <= 0.01) or (beta <= 0.01) or (delta <= 0.01):  # if param goes below 0
            if_adjust = 1
            rv = np.random.randn(len(pv_best))  # radius
            rv = rv / np.linalg.norm(rv)  # normalize to 1
            pv = pv_best + 0.01 * rv  # perturb within a 0.01 radius ball
            alpha, beta, delta = pv
        if if_adjust:
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] ADJUST: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )
            if_adjust = 0

        # logger.info("-" * 110)

    # End optimization & record best
    # logger.info("-" * 110)
    best_results["iteration"].append(iteration_best)
    best_results["R"].append(R_best)
    best_results["Rk"].append(Rk_best)
    best_results["alpha"].append(alpha_best)
    best_results["beta"].append(beta_best)
    best_results["delta"].append(delta_best)
    best_results["sol"].append(sol_best)
    best_results["u0"].append(u0_best)
    logger.info(
        f"BEST@{iteration_best:d}: L2 Residual = {Rk_best:.4f} | alpha = {alpha_best:.4f}   "
        f"beta = {beta_best:.4f}   delta = {delta_best:.4f}"
    )
    # logger.info("*" * 110)
    # logger.info("*" * 110)

    # NOTE: If you want to plot glottal flow, estimatted glottal flow and residual
    # plt.figure()
    # plt.plot(sol_best[:, 0], glottal_flow, "k.-")
    # plt.plot(sol_best[:, 0], u0_best, "b.-")
    # plt.plot(sol_best[:, 0], R_best, "r.-")
    # plt.xlabel("t")
    # plt.legend(["glottal flow", "estimated glottal flow", "residual"])
    # plt.figure()
    # plt.subplot(121)
    # plt.plot(sol_best[:, 1], sol_best[:, 3], "b.-")
    # plt.xlabel(r"$\xi_r$")
    # plt.ylabel(r"$\xi_l$")
    # plt.subplot(122)
    # plt.plot(sol_best[:, 2], sol_best[:, 4], "b.-")
    # plt.xlabel(r"$\dot{\xi}_r$")
    # plt.ylabel(r"$\dot{\xi}_l$")
    # plt.tight_layout()
    # plt.show()

    # logger.info("Saving results")
    return best_results

