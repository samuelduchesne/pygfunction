# -*- coding: utf-8 -*-
""" Test suite for the Laplace-domain g-function solver.

The 'laplace' solver evaluates the exact continuous-time g-function (for
the spatial discretization at hand), while the time-marching solvers use
piecewise-constant heat extraction rates over the time steps. The two are
therefore validated against one another as follows :

- For the 'UHTR' boundary condition, the time-marching solution is exact
  in time : the two solvers must agree tightly.
- For the 'UBWT' boundary condition, the time-marching solution converges
  to the exact continuous-time solution as its time grid is refined : the
  difference to the 'laplace' solver must decrease with grid refinement.
"""
import numpy as np
import pytest

import pygfunction as gt


@pytest.fixture
def small_field():
    return gt.boreholes.rectangle_field(3, 2, 7.5, 7.5, 150., 4., 0.075)


# =============================================================================
# Laplace-domain finite line source transfer function
# =============================================================================
# Test the Laplace-domain FLS transfer function against a numerical Laplace
# transform of the time-domain FLS solution
@pytest.mark.parametrize("dis, H1, D1, H2, D2", [
    (0.075, 150., 4., 150., 4.),
    (7.5, 150., 4., 150., 4.),
    (7.5, 18.75, 41.5, 18.75, 116.5),
    ])
def test_finite_line_source_laplace(dis, H1, D1, H2, D2):
    from scipy.integrate import quad
    from pygfunction.heat_transfer import (
        finite_line_source_vectorized,
        _finite_line_source_laplace,
        _finite_line_source_steady_state,
        )
    alpha = 1e-6
    for p in [1e-11, 1e-9, 3e-8]:
        # Reference : p * integral of h(t) * exp(-p * t) over all times,
        # evaluated in the log-time variable
        def integrand(u):
            t = np.exp(u)
            h = finite_line_source_vectorized(
                float(t), alpha, dis, H1, D1, H2, D2)
            return float(h) * np.exp(-p * t) * p * t
        reference, _ = quad(
            integrand, np.log(1e-2), np.log(1e15 / p), epsabs=1e-13,
            epsrel=1e-11, limit=800)
        h = float(_finite_line_source_laplace(
            p, alpha, dis, H1, D1, H2, D2))
        assert np.isclose(h, reference, rtol=1e-9, atol=1e-12)
    # The transfer function at p = 0 is the steady-state solution
    h_steady = float(_finite_line_source_laplace(
        0., alpha, dis, H1, D1, H2, D2))
    reference = float(_finite_line_source_steady_state(
        dis, H1, D1, H2, D2, True, True))
    assert np.isclose(h_steady, reference, rtol=1e-12)


# =============================================================================
# 'UHTR' boundary condition
# =============================================================================
# The time-marching evaluation of the 'UHTR' g-function is exact in time :
# the 'laplace' solver must reproduce it tightly
def test_gfunction_laplace_UHTR(small_field):
    alpha = 1e-6
    ts = 150.**2 / (9. * alpha)
    time = np.exp(np.linspace(-8.5, 3.4, 12)) * ts
    gFunc_reference = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='similarities',
        boundary_condition='UHTR', options={'nSegments': 8}).gFunc
    gFunc = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='laplace',
        boundary_condition='UHTR', options={'nSegments': 8}).gFunc
    assert np.allclose(gFunc, gFunc_reference, rtol=1e-5)


# =============================================================================
# 'UBWT' boundary condition
# =============================================================================
# The time-marching evaluation of the 'UBWT' g-function converges to the
# (exact in time) 'laplace' solution as the time grid is refined
def test_gfunction_laplace_UBWT_convergence(small_field):
    alpha = 1e-6
    ts = 150.**2 / (9. * alpha)
    time = np.exp(np.linspace(-8.5, 3.4, 12)) * ts
    gFunc_laplace = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='laplace',
        boundary_condition='UBWT', options={'nSegments': 8}).gFunc
    differences = []
    for nt in [24, 96, 384]:
        time_fine = np.unique(np.concatenate(
            (np.exp(np.linspace(np.log(time[0] / 50.), np.log(time[-1]),
                                nt)),
             time)))
        gFunc_fine = gt.gfunction.gFunction(
            small_field, alpha, time=time_fine, method='similarities',
            boundary_condition='UBWT', options={'nSegments': 8}).gFunc
        i_time = np.searchsorted(time_fine, time)
        differences.append(
            np.max(np.abs(gFunc_fine[i_time] - gFunc_laplace)
                   / gFunc_laplace))
    # The time-marching solution approaches the Laplace-domain solution
    assert differences[2] < differences[1] < differences[0]
    assert differences[2] < 2.5e-4


# The 'laplace' solver agrees with the time-marching solution within the
# time-discretization error of the latter
def test_gfunction_laplace_UBWT(small_field):
    alpha = 1e-6
    ts = 150.**2 / (9. * alpha)
    time = np.exp(np.linspace(-8.5, 3.4, 26)) * ts
    gFunc_reference = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='similarities',
        boundary_condition='UBWT', options={'nSegments': 8}).gFunc
    gFunc = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='laplace',
        boundary_condition='UBWT', options={'nSegments': 8}).gFunc
    assert np.allclose(gFunc, gFunc_reference, rtol=5e-3)


# =============================================================================
# Factored (matrix-free) path
# =============================================================================
# The factored path of the 'laplace' solver returns the same g-function as
# its dense path
@pytest.mark.parametrize("boundary_condition", ['UBWT', 'UHTR'])
def test_gfunction_laplace_factored(small_field, boundary_condition):
    alpha = 1e-6
    ts = 150.**2 / (9. * alpha)
    time = np.exp(np.linspace(-8.5, 3.4, 12)) * ts
    solver_dense = gt.solvers.Laplace(
        small_field, None, time, boundary_condition, nSegments=8)
    gFunc_dense = solver_dense.solve(time, alpha)

    # Force the factored path by lowering the size thresholds
    class LaplaceFactored(gt.solvers.Laplace):
        _factored_solver_min_nSources = 1
        _factored_solver_operations_ratio = 0

    solver_factored = LaplaceFactored(
        small_field, None, time, boundary_condition, nSegments=8)
    assert solver_factored._identical_vertical_field
    assert solver_factored._use_factored_solver(len(time))
    gFunc_factored = solver_factored.solve(time, alpha)
    assert np.allclose(gFunc_factored, gFunc_dense, rtol=1e-6)


# Time values of zero (and below) evaluate to a zero g-function and
# infinite time values evaluate to the steady-state g-function
def test_gfunction_laplace_time_edge_cases(small_field):
    from pygfunction.heat_transfer import _finite_line_source_steady_state
    alpha = 1e-6
    ts = 150.**2 / (9. * alpha)
    time = np.array([0., 0.1 * ts, 10. * ts, np.inf])
    gFunc = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='laplace',
        boundary_condition='UBWT', options={'nSegments': 8}).gFunc
    # Zero g-function at t = 0
    assert gFunc[0] == 0.
    # The g-function is increasing and bounded by the steady-state value
    assert np.all(np.diff(gFunc) > 0.)
    # Interior values agree with the same solver on a regular time vector
    gFunc_regular = gt.gfunction.gFunction(
        small_field, alpha, time=np.array([0.1 * ts, 10. * ts]),
        method='laplace', boundary_condition='UBWT',
        options={'nSegments': 8}).gFunc
    assert np.allclose(gFunc[1:3], gFunc_regular, rtol=1e-6)


# =============================================================================
# Heat extraction rate profiles
# =============================================================================
# The heat extraction rate profiles conserve energy at all times
def test_gfunction_laplace_profiles(small_field):
    alpha = 1e-6
    ts = 150.**2 / (9. * alpha)
    time = np.exp(np.linspace(-8.5, 3.4, 12)) * ts
    gFunc = gt.gfunction.gFunction(
        small_field, alpha, time=time, method='laplace',
        boundary_condition='UBWT', options={'nSegments': 8,
                                            'profiles': True})
    solver = gFunc.solver
    H_b = solver.segment_lengths()
    H_tot = np.sum(H_b)
    # Energy conservation : sum(Q_b * H_b) = H_tot at all times
    assert np.allclose(H_b @ solver.Q_b / H_tot, 1., rtol=1e-4)
    # The borehole wall temperature is the g-function
    assert np.allclose(solver.T_b, gFunc.gFunc)
