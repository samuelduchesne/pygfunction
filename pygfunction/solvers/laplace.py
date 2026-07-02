# -*- coding: utf-8 -*-
import warnings
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear

from .similarities import Similarities
from ..heat_transfer import _finite_line_source_laplace


class Laplace(Similarities):
    """
    Laplace-domain solver for the evaluation of the g-function.

    This solver evaluates the g-function in the Laplace domain, where the
    temporal convolution of the heat extraction rates with the finite line
    source (FLS) solution reduces to an algebraic product. The g-function
    is obtained by sampling the Laplace-domain response at a set of
    (real and positive) values of the Laplace parameter -- each sample is
    an independent, history-free system of equations -- and reconstructing
    the time-domain response as a sum of decaying exponentials:

        .. math::
            g(t) = g_\\infty - \\sum_m w_m e^{-\\lambda_m t},
            \\qquad w_m \\geq 0

    where :math:`g_\\infty` is the (closed-form) steady-state g-function.
    This representation is exact in the limit of infinitely many terms :
    the g-function is the step response of a passive, reciprocal
    (self-adjoint) diffusion system, so that its time derivative is
    completely monotone (Bernstein's theorem). The weights are identified
    by regularized least squares on the Laplace-domain samples, with a
    non-negative fit (which guarantees a monotone and bounded
    reconstruction) as a fallback.

    No linearization is applied at small time values : the g-function is
    evaluated exactly at all requested times (it is zero at t <= 0 and
    equal to the steady-state g-function at t = inf).

    In contrast with the time-marching solvers, the solution at each
    Laplace node is independent (no load history reconstruction and no
    temporal superposition), the number of systems of equations is
    independent of the number of time values, and the resulting
    exponential sum evaluates the g-function at any time.

    The solver only supports vertical boreholes and the 'UHTR' and 'UBWT'
    boundary conditions. The `kind` (interpolation) and `approximate_FLS`
    options are not used by this solver.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    network : network object
        Model of the network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be
        one of

            - 'UHTR' :
                **Uniform heat transfer rate**.
            - 'UBWT' :
                **Uniform borehole wall temperature**.

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is
        provided, it must return an array of size (nSegments,) when
        provided with nSegments (of type int) as an argument, or an array
        of size (nSegments[i],) when provided with an element of nSegments
        (of type list).
        Default is :func:`utilities.segment_ratios`.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    profiles : bool, optional
        Set to true to keep in memory the temperatures and heat extraction
        rates.
        Default is False.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below
        tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    nodes_per_decade : int, optional
        Number of Laplace-domain sample nodes per decade of the Laplace
        parameter.
        Default is 8.
    modes_per_decade : int, optional
        Number of candidate decay rates per decade for the exponential-sum
        reconstruction of the g-function.
        Default is 8.

    """
    # Margin factors for the range of the Laplace parameter relative to
    # the reciprocal of the requested time values
    _p_min_factor = 0.03
    _p_max_factor = 30.

    def initialize(self, nodes_per_decade=8, modes_per_decade=8, **kwargs):
        self.nodes_per_decade = nodes_per_decade
        self.modes_per_decade = modes_per_decade
        return super().initialize(**kwargs)

    def solve(self, time, alpha):
        """
        Build and solve the system of equations in the Laplace domain.

        Parameters
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is
            evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        gFunc : float or array
            Values of the g-function.

        """
        # Number of time values
        self.time = time
        t = np.atleast_1d(np.asarray(self.time, dtype=float))
        # Time values of zero (or below) evaluate to a zero g-function and
        # infinite time values evaluate to the steady-state g-function :
        # only positive finite time values set the range of the Laplace
        # nodes
        t_positive = t[(t > 0.) & np.isfinite(t)]
        if len(t_positive) > 0:
            t_min = np.min(t_positive)
            t_max = np.max(t_positive)
        else:
            # Degenerate time vector : only the steady-state solution and
            # the zero solution are required. A dummy range is used.
            t_min = 1.
            t_max = 1.
        # Laplace-domain sample nodes (log-spaced, with the steady-state
        # solution at p = 0 as the first node)
        p_min = self._p_min_factor / t_max
        p_max = self._p_max_factor / t_min
        nP = max(
            int(np.ceil(
                np.log10(p_max / p_min) * self.nodes_per_decade)) + 1,
            2)
        p_nodes = np.concatenate(
            ([0.], np.logspace(np.log10(p_min), np.log10(p_max), nP)))

        # Laplace-domain transfer matrices at all sample nodes
        G, Q_transfer = self._laplace_domain_response(p_nodes, alpha)

        if self.disp:
            print('Reconstructing the g-function ...', end='')
        tic = perf_counter()
        # Steady-state g-function (the p = 0 sample)
        g_infinity = G[0]
        # Exponential-sum reconstruction of the g-function :
        #     g(t) = g_infinity - sum_m w_m * exp(-lambda_m * t)
        lambdas, w = self._fit_exponential_sum(
            p_nodes[1:], G[1:], g_infinity, t_min, t_max)
        decay = np.exp(-np.multiply.outer(t, lambdas))
        gFunc = g_infinity - decay @ w
        # The g-function is zero at (and below) time t=0. Infinite time
        # values are correctly evaluated by the exponential sum (the decay
        # factors vanish).
        gFunc[t <= 0.] = 0.

        # Store temperature and heat extraction rate profiles
        if self.profiles:
            c = self._fit_profiles(p_nodes[1:], Q_transfer, lambdas)
            profiles = Q_transfer[:, 0:1] - c @ decay.T
            if self.boundary_condition == 'UHTR':
                # For 'UHTR', the transfer functions hold the borehole
                # wall temperatures under uniform heat extraction rates
                profiles[:, t <= 0.] = 0.
                self.Q_b = 1
                self.T_b = profiles
            else:
                profiles[:, t <= 0.] = 1.
                self.Q_b = profiles
                self.T_b = gFunc
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')
        return gFunc

    def _laplace_domain_response(self, p_nodes, alpha):
        """
        Evaluate the Laplace-domain response at all sample nodes.

        For the 'UBWT' boundary condition, the system of equations:

            [K(p)] @ [Q'] = [1] * G(p),   [H_b] @ [Q'] = H_tot

        is solved at each node, where [K(p)] is the matrix of Laplace-
        domain transfer functions of the thermal response factors,
        [Q'] = p * [Q_hat] is the transfer function of the heat extraction
        rates and G(p) = p * T_hat_b is the transfer function of the
        borehole wall temperature. This system is identical in form to the
        the time-domain system at a single time step with no load history :
        the existing solution methods are reused. For the 'UHTR' boundary
        condition, G(p) follows directly from the row sums of [K(p)].

        Parameters
        ----------
        p_nodes : array
            Values of the Laplace parameter (in 1/seconds), with the
            steady-state node p = 0 first.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        G : array
            Transfer function of the g-function at all sample nodes.
        Q_transfer : array
            Transfer functions of the segment heat extraction rates at all
            sample nodes, shape (nSources, nP).

        """
        nP = len(p_nodes)
        H_b = self.segment_lengths()
        H_tot = np.sum(H_b)
        G = np.empty(nP)
        Q_transfer = np.empty((self.nSources, nP))
        # The factored (matrix-free) path is used under the same conditions
        # as the time-domain solver
        nb = len(self.boreholes)
        use_factored = (
            self._identical_vertical_field and nb > 1
            and self._use_factored_solver(nP))
        if use_factored:
            if self.disp:
                print('Calculating Laplace-domain response factors ...',
                      end='')
            tic = perf_counter()
            B_self, B_dis, Adj, D_idx = self._factored_response_factors(
                p_nodes, alpha)
            toc = perf_counter()
            if self.disp: print(f' {toc - tic:.3f} sec')
            if self.disp:
                print('Solving the systems of equations ...', end='')
            tic = perf_counter()
            H_seg = H_b[0:self.nBoreSegments[0]]
            T_b0 = np.zeros(self.nSources)
            X_1 = None
            X_2 = None
            if self.boundary_condition == 'UHTR':
                # Row sums of the transfer matrices
                ones = np.reshape(
                    np.ones(self.nSources), (nb, self.nBoreSegments[0]))
                nDis = B_dis.shape[1]
                Z = (Adj @ ones).reshape(nDis, nb, -1)
                for m in range(nP):
                    T = np.einsum(
                        'dmn,dbn->bm', B_dis[m+1], Z) + ones @ B_self[m+1].T
                    Q_transfer[:, m] = T.flatten()
                    G[m] = H_b @ Q_transfer[:, m] / H_tot
            else:
                for m in range(nP):
                    Q_transfer[:, m], G[m], X_1, X_2 = \
                        self._factored_solve_step(
                            Adj, D_idx, B_self[m+1], B_dis[m+1], T_b0,
                            H_seg, H_b, H_tot, X_1, X_2)
            toc = perf_counter()
            if self.disp: print(f' {toc - tic:.3f} sec')
        else:
            # Dense transfer matrices at all sample nodes, assembled with
            # the same machinery as the time-domain thermal response
            # factors. Only the data of the returned interpolator is used
            # (the leading matrix along the last axis corresponds to
            # "time zero" and is discarded).
            K = self.thermal_response_factors(p_nodes, alpha).y[:, :, 1:]
            if self.disp:
                print('Solving the systems of equations ...', end='')
            tic = perf_counter()
            if self.boundary_condition == 'UHTR':
                # For 'UHTR', the transfer functions of the borehole wall
                # temperatures are the row sums of the transfer matrices
                Q_transfer = np.sum(K, axis=1)
                G = H_b @ Q_transfer / H_tot
            else:
                T_b0 = np.zeros(self.nSources)
                for m in range(nP):
                    Q_transfer[:, m], G[m] = \
                        self._solve_uniform_borehole_wall_temperature(
                            K[:, :, m], T_b0, H_b, H_tot)
            toc = perf_counter()
            if self.disp: print(f' {toc - tic:.3f} sec')
        return G, Q_transfer

    def _fit_exponential_sum(self, p, G, g_infinity, t_min, t_max):
        """
        Fit an exponential-sum representation of the g-function from its
        Laplace-domain samples.

        The g-function is represented as:

            g(t) = g_infinity - sum_m w_m * exp(-lambda_m * t)

        which corresponds, in the Laplace domain, to:

            g_infinity - G(p) = sum_m w_m * p / (p + lambda_m)

        The weights are identified by (truncated-SVD-regularized) least
        squares over a log-spaced grid of candidate decay rates lambda_m.
        The reconstruction is verified to be monotone and bounded ;
        otherwise, a non-negative fit (justified by the complete
        monotonicity of the time derivative of the g-function, and
        guaranteeing admissibility) is used as a fallback.

        Parameters
        ----------
        p : array
            Values of the Laplace parameter (in 1/seconds).
        G : array
            Transfer function of the g-function at the sample nodes.
        g_infinity : float
            Steady-state g-function.
        t_min : float
            Minimum time (in seconds) at which the g-function is evaluated.
        t_max : float
            Maximum time (in seconds) at which the g-function is evaluated.

        Returns
        -------
        lambdas : array
            Decay rates (in 1/seconds) of the exponential-sum
            representation.
        w : array
            Weights of the exponential-sum representation.

        """
        # Candidate decay rates (log-spaced)
        lambda_min = self._p_min_factor / (3. * t_max)
        lambda_max = 3. * self._p_max_factor / t_min
        nModes = max(
            int(np.ceil(
                np.log10(lambda_max / lambda_min)
                * self.modes_per_decade)) + 1,
            2)
        lambdas = np.logspace(
            np.log10(lambda_min), np.log10(lambda_max), nModes)
        # Least squares on the Laplace-domain samples. The
        # (truncated-SVD-regularized) unconstrained problem is solved
        # first : it is more accurate than constrained solvers on the
        # highly collinear design matrix. The reconstruction is verified
        # to be physically admissible (non-decreasing and bounded by the
        # steady-state value) ; otherwise, a non-negative fit (which
        # guarantees admissibility) is used instead.
        A = _exponential_sum_design_matrix(p, lambdas)
        y = g_infinity - G

        def is_admissible(w):
            t_check = np.logspace(np.log10(t_min), np.log10(t_max), 200)
            g_check = g_infinity \
                - np.exp(-np.multiply.outer(t_check, lambdas)) @ w
            tolerance = 1e-6 * max(abs(g_infinity), 1.)
            return (np.all(np.diff(g_check) >= -tolerance)
                    and np.all(g_check >= -tolerance)
                    and np.all(g_check <= g_infinity + tolerance))

        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        if not is_admissible(w):
            # Fall back to a non-negative fit, which guarantees a monotone
            # and bounded reconstruction
            warnings.warn(
                'The exponential-sum reconstruction of the g-function was '
                'not monotone and bounded : a non-negative fit is used '
                'instead. The g-function accuracy may be reduced.',
                RuntimeWarning)
            result = lsq_linear(
                A, y, bounds=(0., np.inf), method='bvls', tol=1e-14,
                max_iter=10*len(lambdas))
            w = result.x
            if not result.success:
                warnings.warn(
                    'The non-negative exponential-sum fit did not '
                    'converge. The g-function accuracy may be reduced.',
                    RuntimeWarning)
        return lambdas, w

    def _fit_profiles(self, p, Q_transfer, lambdas):
        """
        Fit exponential-sum representations of the segment heat extraction
        rates using the decay rates identified for the g-function.

        Parameters
        ----------
        p : array
            Values of the Laplace parameter (in 1/seconds), excluding the
            steady-state node.
        Q_transfer : array
            Transfer functions of the segment heat extraction rates at all
            sample nodes (with the steady-state node first).
        lambdas : array
            Decay rates (in 1/seconds) of the exponential-sum
            representation.

        Returns
        -------
        c : array
            Weights of the exponential-sum representations, shape
            (nSources, nModes).

        """
        A = _exponential_sum_design_matrix(p, lambdas)
        y = Q_transfer[:, 0:1] - Q_transfer[:, 1:]
        c, *_ = np.linalg.lstsq(A, y.T, rcond=None)
        return c.T

    def _fls_kernel(self, time, alpha, dis, H1, D1, H2, D2):
        """
        Evaluate the Laplace-domain transfer function of the FLS solution.

        The `time` argument holds values of the Laplace parameter (in
        1/seconds). The node p = 0 evaluates the steady-state FLS solution.

        """
        return _finite_line_source_laplace(
            time, alpha, dis, H1, D1, H2, D2)

    def _check_solver_specific_inputs(self):
        """
        This method ensures that solver specific inputs to the Solver
        object are what is expected.

        """
        super()._check_solver_specific_inputs()
        assert self.boundary_condition in ('UHTR', 'UBWT'), \
            "The 'laplace' solver only supports the 'UHTR' and 'UBWT' " \
            "boundary conditions. Note that the boundary condition " \
            "defaults to 'MIFT' when a network is provided : provide " \
            "boundary_condition='UBWT' (or 'UHTR') explicitly to use " \
            "the 'laplace' solver."
        assert not np.any([b.is_tilted() for b in self.boreholes]), \
            "The 'laplace' solver only supports vertical boreholes."
        assert type(self.nodes_per_decade) is int \
            and self.nodes_per_decade >= 2, \
            "The option 'nodes_per_decade' should be an int greater or " \
            "equal to 2."
        assert type(self.modes_per_decade) is int \
            and self.modes_per_decade >= 2, \
            "The option 'modes_per_decade' should be an int greater or " \
            "equal to 2."
        return


def _exponential_sum_design_matrix(p, lambdas):
    """
    Design matrix of the exponential-sum collocation problem.

    The Laplace transform of a sum of decaying exponentials gives, for the
    transfer function samples:

        y(p) = sum_m w_m * p / (p + lambda_m)

    Parameters
    ----------
    p : array
        Values of the Laplace parameter (in 1/seconds), shape (nP,).
    lambdas : array
        Candidate decay rates (in 1/seconds), shape (nModes,).

    Returns
    -------
    A : array
        Design matrix, shape (nP, nModes).

    """
    return p[:, np.newaxis] / (p[:, np.newaxis] + lambdas)
