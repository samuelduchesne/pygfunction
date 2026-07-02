# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
from scipy.special import erfc, erf, roots_legendre

from .boreholes import Borehole
from .utilities import erfint, exp1, _erf_coeffs


def finite_line_source(
        time, alpha, borehole1, borehole2, reaSource=True, imgSource=True,
        approximation=False, M=11, N=10):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution. For vertical boreholes, the FLS solution was proposed
    by Claesson and Javed [#FLS-ClaJav2011]_ and extended to boreholes with
    different vertical positions by Cimmino and Bernier [#FLS-CimBer2014]_.
    The FLS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            e^{-d_{12}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            d_{12} &= \\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}


            I_{real}(s) &= erfint((D_2-D_1+H_2)s) - erfint((D_2-D_1)s)

            &+ erfint((D_2-D_1-H_1)s) - erfint((D_2-D_1+H_2-H_1)s)

            I_{imag}(s) &= erfint((D_2+D_1+H_2)s) - erfint((D_2+D_1)s)

            &+ erfint((D_2+D_1+H_1)s) - erfint((D_2+D_1+H_2+H_1)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

    For inclined boreholes, the FLS solution was proposed by Lazzarotto
    [#FLS-Lazzar2016]_ and Lazzarotto and Björk [#FLS-LazBjo2016]_.
    The FLS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{H_1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            \\frac{1}{s}
            \\int_{0}^{1} (I_{real}(u, s)+I_{imag}(u, s)) du ds


            I_{real}(u, s) &=
            e^{-((x_1 - x_2)^2 + (y_1 - y_2)^2 + (D_1 - D_2)^2) s^2}

            &\\cdot (erf((u H_1 k_{0,real} + k_{2,real}) s)
             - erf((u H_1 k_{0,real} + k_{2,real} - H_2) s))

            &\\cdot  e^{(u^2 H_1^2 (k_{0,real}^2 - 1)
                + 2 u H_1 (k_{0,real} k_{2,real} - k_{1,real}) + k_{2,real}^2) s^2}
            du ds


            I_{imag}(u, s) &=
            -e^{-((x_1 - x_2)^2 + (y_1 - y_2)^2 + (D_1 + D_2)^2) s^2}

            &\\cdot (erf((u H_1 k_{0,imag} + k_{2,imag}) s)
             - erf((u H_1 k_{0,imag} + k_{2,imag} - H_2) s))

            &\\cdot e^{(u^2 H_1^2 (k_{0,imag}^2 - 1)
                + 2 u H_1 (k_{0,imag} k_{2,imag} - k_1) + k_{2,imag}^2) s^2}
            du ds


            k_{0,real} &=
            sin(\\beta_1) sin(\\beta_2) cos(\\theta_1 - \\theta_2)
            + cos(\\beta_1) cos(\\beta_2)


            k_{0,imag} &=
            sin(\\beta_1) sin(\\beta_2) cos(\\theta_1 - \\theta_2)
            - cos(\\beta_1) cos(\\beta_2)


            k_{1,real} &= sin(\\beta_1)
            (cos(\\theta_1) (x_1 - x_2) + sin(\\theta_1) (y_1 - y_2))
            + cos(\\beta_1) (D_1 - D_2)


            k_{1,imag} &= sin(\\beta_1)
            (cos(\\theta_1) (x_1 - x_2) + sin(\\theta_1) (y_1 - y_2))
            + cos(\\beta_1) (D_1 + D_2)


            k_{2,real} &= sin(\\beta_2)
            (cos(\\theta_2) (x_1 - x_2) + sin(\\theta_2) (y_1 - y_2))
            + cos(\\beta_2) (D_1 - D_2)


            k_{2,imag} &= sin(\\beta_2)
            (cos(\\theta_2) (x_1 - x_2) + sin(\\theta_2) (y_1 - y_2))
            - cos(\\beta_2) (D_1 + D_2)

    where :math:`\\beta_1` and :math:`\\beta_2` are the tilt angle of the
    boreholes (relative to vertical), and :math:`\\theta_1` and
    :math:`\\theta_2` are the orientation of the boreholes (relative to the
    x-axis).

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    borehole1 : Borehole object or list of Borehole objects, length (N)
        Borehole object of the borehole extracting heat.
    borehole2 : Borehole object or list of Borehole objects, length (M)
        Borehole object for which the FLS is evaluated.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is True.
    approximation : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#FLS-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    M : int, optional
        Number of Gauss-Legendre sample points for the quadrature over
        :math:`u`. This is only used for inclined boreholes.
        Default is 11.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float or array, shape (M, N, K), (M, N) or (K)
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    The function returns a float if time is a float and borehole1 and borehole2
    are Borehole objects. If time is a float and any of borehole1 and borehole2
    are lists, the function returns an array, shape (M, N), If time is an array
    and borehole1 and borehole2 are Borehole objects, the function returns an
    array, shape (K).If time is an array and any of borehole1 and borehole2 are
    are lists, the function returns an array, shape (M, N, K).

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> h = gt.heat_transfer.finite_line_source(4*168*3600., 1.0e-6, b1, b2)
    h = 0.0110473635393
    >>> h = gt.heat_transfer.finite_line_source(
        4*168*3600., 1.0e-6, b1, b2, approximation=True, N=10)
    h = 0.0110474667731
    >>> b3 = gt.boreholes.Borehole(
        H=150., D=4., r_b=0.075, x=5., y=0., tilt=3.1415/15, orientation=0.)
    >>> h = gt.heat_transfer.finite_line_source(
        4*168*3600., 1.0e-6, b1, b3, M=21)
    h = 0.0002017450051

    References
    ----------
    .. [#FLS-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLS-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#FLS-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#FLS-Lazzar2016] Lazzarotto, A. (2016). A methodology for the
       calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 1, Renewable Energy, 86,
       1380-1393.
    .. [#FLS-LazBjo2016] Lazzarotto, A., & Björk, F. (2016). A methodology for
       the calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 2, Renewable Energy, 86,
       1353-1361.

    """
    if isinstance(borehole1, Borehole) and isinstance(borehole2, Borehole):
        # Unpack parameters
        H1, D1 = borehole1.H, borehole1.D
        H2, D2 = borehole2.H, borehole2.D
        if borehole1.is_vertical() and borehole2.is_vertical():
            # Boreholes are vertical
            dis = borehole1.distance(borehole2)
            if time is np.inf:
                # Steady-state solution
                h = _finite_line_source_steady_state(
                    dis, H1, D1, H2, D2, reaSource, imgSource)
            elif approximation:
                # Approximation
                h = finite_line_source_approximation(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource, N=N)
            else:
                # Evaluate integral
                h = finite_line_source_vectorized(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource)
        else:
            # At least one borehole is tilted
            # Unpack parameters
            x1, y1 = borehole1.position()
            rb1 = borehole1.r_b
            tilt1 = borehole1.tilt
            orientation1 = borehole1.orientation
            x2, y2 = borehole2.position()
            tilt2 = borehole2.tilt
            orientation2 = borehole2.orientation
            if time is np.inf:
                # Steady-state solution
                h = _finite_line_source_inclined_steady_state(
                    rb1, x1, y1, H1, D1, tilt1, orientation1,
                    x2, y2, H2, D2, tilt2, orientation2,
                    reaSource, imgSource, M=M)
            elif approximation:
                # Approximation
                h = finite_line_source_inclined_approximation(
                    time, alpha, rb1, x1, y1, H1, D1, tilt1, orientation1,
                    x2, y2, H2, D2, tilt2, orientation2,
                    reaSource=reaSource, imgSource=imgSource, M=M, N=N)
            else:
                # Evaluate integral
                h = finite_line_source_inclined_vectorized(
                    time, alpha,
                    rb1, x1, y1, H1, D1, tilt1, orientation1,
                    x2, y2, H2, D2, tilt2, orientation2,
                    reaSource=reaSource, imgSource=imgSource, M=M)

    else:
        # Unpack parameters
        if isinstance(borehole1, Borehole): borehole1 = [borehole1]
        if isinstance(borehole2, Borehole): borehole2 = [borehole2]
        x1 = np.array([b.x for b in borehole1])
        y1 = np.array([b.y for b in borehole1])
        x2 = np.array([b.x for b in borehole2])
        y2 = np.array([b.y for b in borehole2])
        r_b = np.array([b.r_b for b in borehole1])
        dis = np.maximum(
            np.sqrt(np.add.outer(x2, -x1)**2 + np.add.outer(y2, -y1)**2),
            r_b)
        D1 = np.array([b.D for b in borehole1]).reshape(1, -1)
        H1 = np.array([b.H for b in borehole1]).reshape(1, -1)
        D2 = np.array([b.D for b in borehole2]).reshape(-1, 1)
        H2 = np.array([b.H for b in borehole2]).reshape(-1, 1)

        if (np.all([b.is_vertical() for b in borehole1])
            and np.all([b.is_vertical() for b in borehole2])):
            # All boreholes are vertical
            if time is np.inf:
                # Steady-state solution
                h = _finite_line_source_steady_state(
                    dis, H1, D1, H2, D2, reaSource, imgSource)
            elif approximation:
                # Approximation
                h = finite_line_source_approximation(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource, N=N)
            else:
                # Evaluate integral
                h = finite_line_source_vectorized(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource)
        else:
            # At least one borehole is tilted
            # Unpack parameters
            x1 = x1.reshape(1, -1)
            y1 = y1.reshape(1, -1)
            tilt1 = np.array([b.tilt for b in borehole1]).reshape(1, -1)
            orientation1 = np.array([b.orientation for b in borehole1]).reshape(1, -1)
            x2 = x2.reshape(-1, 1)
            y2 = y2.reshape(-1, 1)
            tilt2 = np.array([b.tilt for b in borehole2]).reshape(-1, 1)
            orientation2 = np.array([b.orientation for b in borehole2]).reshape(-1, 1)
            r_b = r_b.reshape(1, -1)
            if time is np.inf:
                # Steady-state solution
                h = _finite_line_source_inclined_steady_state(
                    r_b, x1, y1, H1, D1, tilt1, orientation1,
                    x2, y2, H2, D2, tilt2, orientation2,
                    reaSource, imgSource, M=M)
            elif approximation:
                # Approximation
                h = finite_line_source_inclined_approximation(
                    time, alpha, r_b, x1, y1, H1, D1, tilt1, orientation1,
                    x2, y2, H2, D2, tilt2, orientation2,
                    reaSource=reaSource, imgSource=imgSource, M=M, N=N)
            else:
                # Evaluate integral
                h = finite_line_source_inclined_vectorized(
                    time, alpha,
                    r_b, x1, y1, H1, D1, tilt1, orientation1,
                    x2, y2, H2, D2, tilt2, orientation2,
                    reaSource=reaSource, imgSource=imgSource, M=M)
    return h


def finite_line_source_approximation(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True,
        N=10):
    """
    Evaluate the Finite Line Source (FLS) solution using the approximation
    of Cimmino (2021) [#FLSApprox-Cimmin2021]_.

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool, optional
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is True.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h


    References
    ----------
    .. [#FLSApprox-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """

    dis = np.divide.outer(dis, np.sqrt(4*alpha*time))
    H1 = np.divide.outer(H1, np.sqrt(4*alpha*time))
    D1 = np.divide.outer(D1, np.sqrt(4*alpha*time))
    H2 = np.divide.outer(H2, np.sqrt(4*alpha*time))
    D2 = np.divide.outer(D2, np.sqrt(4*alpha*time))
    p, q = _finite_line_source_coefficients(
        H1, D1, H2, D2, reaSource, imgSource)
    q = np.abs(q)
    # Coefficients of the approximation of the error function
    a, b = _erf_coeffs(N)

    dd = dis**2
    qq = q**2
    G1 = np.inner(
        p,
        q * np.inner(
            a,
            0.5*  exp1(np.expand_dims(dd, axis=(-1, -2)) + np.multiply.outer(qq, b))))
    x3 = np.sqrt(np.expand_dims(dd, axis=-1) + qq)
    G3 = np.inner(p, np.exp(-x3**2) / np.sqrt(np.pi) - x3 * erfc(x3))

    h = 0.5 / H2 * (G1 + G3)
    return h


def finite_line_source_inclined_approximation(
        time, alpha,
        rb1, x1, y1, H1, D1, tilt1, orientation1,
        x2, y2, H2, D2, tilt2, orientation2,
        reaSource=True, imgSource=True, M=11, N=10):
    """
    Evaluate the inclined Finite Line Source (FLS) solution using the
    approximation method of Cimmino (2021) [#IncFLSApprox-Cimmin2021]_.

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    rb1 : array
        Radii of the emitting heat sources.
    x1 : float or array
        x-Positions of the emitting heat sources.
    y1 : float or array
        y-Positions of the emitting heat sources.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    tilt1 : float or array
        Angles (in radians) from vertical of the emitting heat sources.
    orientation1 : float or array
        Directions (in radians) of the tilt the emitting heat sources.
    x2 : array
        x-Positions of the receiving heat sources.
    y2 : array
        y-Positions of the receiving heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    tilt2 : float or array
        Angles (in radians) from vertical of the receiving heat sources.
    orientation2 : float or array
        Directions (in radians) of the tilt the receiving heat sources.
    reaSource : bool, optional
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is true.
    M : int, optional
        Number of points for the Gauss-Legendre quadrature rule along the
        receiving heat sources.
        Default is 21.
    N : int, optional
        Number of terms in the approximation of the FLS solution.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    References
    ----------
    .. [#IncFLSApprox-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    # Expected output shape of h, excluding time
    output_shape = np.broadcast_shapes(
            *[np.shape(arg) for arg in (
                rb1, x1, y1, H1, D1, tilt1, orientation1,
                x2, y2, H2, D2, tilt2, orientation2)])
    # Number of dimensions of the output, excluding time
    output_ndim = len(output_shape)
    # Shape of the time variable
    time_shape = np.shape(time)
    # Number of dimensions of the time variable
    time_ndim = len(time_shape)
    # Roots for Gauss-Legendre quadrature
    x, w = _roots_legendre_cached(M)
    u = (0.5 * x + 0.5).reshape((-1, 1) + (1,) * output_ndim)
    w = w / 2
    # Coefficients of the approximation of the error function
    a, b = _erf_coeffs(N)
    b = b.reshape((1, -1) + (1,) * output_ndim)
    # Sines and cosines of tilt (b: beta) and orientation (t: theta)
    sb1 = np.sin(tilt1)
    sb2 = np.sin(tilt2)
    cb1 = np.cos(tilt1)
    cb2 = np.cos(tilt2)
    st1 = np.sin(orientation1)
    st2 = np.sin(orientation2)
    ct1 = np.cos(orientation1)
    ct2 = np.cos(orientation2)
    ct12 = np.cos(orientation1 - orientation2)
    # Horizontal distances
    dx = x1 - x2
    dy = y1 - y2
    rr = dx**2 + dy**2  # Squared radial distance
    # Length ratios
    H_ratio = H1 / H2
    H_ratio = np.reshape(H_ratio, np.shape(H_ratio) + (1,) * time_ndim)
    # Approximation
    ss = 1. / (4 * alpha * time)
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        # Axial distances
        dzRea = D1 - D2
        dzImg = D1 + D2
        # FLS-inclined coefficients
        kRea_0 = sb1 * sb2 * ct12 + cb1 * cb2
        kImg_0 = sb1 * sb2 * ct12 - cb1 * cb2
        kRea_1 = sb1 * (ct1 * dx + st1 * dy) + cb1 * dzRea
        kImg_1 = sb1 * (ct1 * dx + st1 * dy) + cb1 * dzImg
        kRea_2 = sb2 * (ct2 * dx + st2 * dy) + cb2 * dzRea
        kImg_2 = sb2 * (ct2 * dx + st2 * dy) - cb2 * dzImg
        dRea_1 = u * H1 * kRea_0 + kRea_2
        dImg_1 = u * H1 * kImg_0 + kImg_2
        dRea_2 = u * H1 * kRea_0 + kRea_2 - H2
        dImg_2 = u * H1 * kImg_0 + kImg_2 - H2
        cRea = np.maximum(
            rr + dzRea**2 - dRea_1**2 + (kRea_1 + u * H1)**2 - kRea_1**2,
            rb1**2)
        cImg = np.maximum(
            rr + dzImg**2 - dImg_1**2 + (kImg_1 + u * H1)**2 - kImg_1**2,
            rb1**2)
        # Signs for summation
        pRea_1 = np.sign(dRea_1)
        pRea_1 = np.reshape(pRea_1, np.shape(pRea_1) + (1,) * time_ndim)
        pRea_2 = np.sign(dRea_2)
        pRea_2 = np.reshape(pRea_2, np.shape(pRea_2) + (1,) * time_ndim)
        pImg_1 = np.sign(dImg_1)
        pImg_1 = np.reshape(pImg_1, np.shape(pImg_1) + (1,) * time_ndim)
        pImg_2 = np.sign(dImg_2)
        pImg_2 = np.reshape(pImg_2, np.shape(pImg_2) + (1,) * time_ndim)
        # FLS-inclined approximation
        h = 0.25 * H_ratio * np.einsum('i,j,ij...', w, a,
            (pRea_1 * exp1(np.multiply.outer(cRea + b * dRea_1**2, ss)) \
            - pRea_2 * exp1(np.multiply.outer(cRea + b * dRea_2**2, ss)) \
            - pImg_1 * exp1(np.multiply.outer(cImg + b * dImg_1**2, ss)) \
            + pImg_2 * exp1(np.multiply.outer(cImg + b * dImg_2**2, ss))) )
    elif reaSource:
        # Real FLS solution
        # Axial distance
        dzRea = D1 - D2
        # FLS-inclined coefficients
        kRea_0 = sb1 * sb2 * ct12 + cb1 * cb2
        kRea_1 = sb1 * (ct1 * dx + st1 * dy) + cb1 * dzRea
        kRea_2 = sb2 * (ct2 * dx + st2 * dy) + cb2 * dzRea
        dRea_1 = u * H1 * kRea_0 + kRea_2
        dRea_2 = u * H1 * kRea_0 + kRea_2 - H2
        cRea = np.maximum(
            rr + dzRea**2 - dRea_1**2 + (kRea_1 + u * H1)**2 - kRea_1**2,
            rb1**2)
        # Signs for summation
        pRea_1 = np.sign(dRea_1)
        pRea_1 = np.reshape(pRea_1, np.shape(pRea_1) + (1,) * time_ndim)
        pRea_2 = np.sign(dRea_2)
        pRea_2 = np.reshape(pRea_2, np.shape(pRea_2) + (1,) * time_ndim)
        # FLS-inclined approximation
        h = 0.25 * H_ratio * np.einsum('i,j,ij...', w, a,
            (pRea_1 * exp1(np.multiply.outer(cRea + b * dRea_1**2, ss)) \
            - pRea_2 * exp1(np.multiply.outer(cRea + b * dRea_2**2, ss))) )
    elif imgSource:
        # Image FLS solution
        # Axial distance
        dzImg = D1 + D2
        # FLS-inclined coefficients
        kImg_0 = sb1 * sb2 * ct12 - cb1 * cb2
        kImg_1 = sb1 * (ct1 * dx + st1 * dy) + cb1 * dzImg
        kImg_2 = sb2 * (ct2 * dx + st2 * dy) - cb2 * dzImg
        dImg_1 = u * H1 * kImg_0 + kImg_2
        dImg_2 = u * H1 * kImg_0 + kImg_2 - H2
        cImg = np.maximum(
            rr + dzImg**2 - dImg_1**2 + (kImg_1 + u * H1)**2 - kImg_1**2,
            rb1**2)
        # Signs for summation
        pImg_1 = np.sign(dImg_1)
        pImg_1 = np.reshape(pImg_1, np.shape(pImg_1) + (1,) * time_ndim)
        pImg_2 = np.sign(dImg_2)
        pImg_2 = np.reshape(pImg_2, np.shape(pImg_2) + (1,) * time_ndim)
        # FLS-inclined approximation
        h = 0.25 * H_ratio * np.einsum('i,j,ij...', w, a,
            (-pImg_1 * exp1(np.multiply.outer(cImg + b * dImg_1**2, ss)) \
            + pImg_2 * exp1(np.multiply.outer(cImg + b * dImg_2**2, ss))) )
    else:
        # No heat source
        h = np.zeros(output_shape + np.shape(time))
    return h


def finite_line_source_vectorized(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True,
        approximation=False, N=10):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Claesson and Javed
    [#FLSVec-ClaJav2011]_ and extended to boreholes with different vertical
    positions by Cimmino and Bernier [#FLSVec-CimBer2014]_. The FLS solution
    is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            e^{-d_{12}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            I_{real}(s) &= erfint((D_2-D_1+H_2)s) - erfint((D_2-D_1)s)

            &+ erfint((D_2-D_1-H_1)s) - erfint((D_2-D_1+H_2-H_1)s)

            I_{imag}(s) &= erfint((D_2+D_1+H_2)s) - erfint((D_2+D_1)s)

            &+ erfint((D_2+D_1+H_1)s) - erfint((D_2+D_1+H_2+H_1)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.
    approximation : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#FLSVec-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    This is a vectorized version of the :func:`finite_line_source` function
    using a fixed Gauss-Legendre quadrature over log-spaced panels to
    evaluate the integrals at all time values simultaneously. All arrays
    (dis, H1, D1, H2, D2) must follow numpy array broadcasting rules. If time
    is an array, the integrals for different time values are stacked on the
    last axis.


    References
    ----------
    .. [#FLSVec-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLSVec-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#FLSVec-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    if not approximation:
        # Integrand of the finite line source solution
        f = _finite_line_source_node_integrand(
            dis, H1, D1, H2, D2, reaSource, imgSource)
        # Steady-state value of the integral (for infinite time values)
        steady_state = lambda: 2. * H2 * _finite_line_source_steady_state(
            dis, H1, D1, H2, D2, reaSource, imgSource)
        # Evaluate integral for all time values
        d_min = float(np.min(dis)) if np.size(dis) > 0 else 1.
        h = _finite_line_source_integral_all_times(
            f, time, alpha, d_min, steady_state=steady_state)
        if _time_is_scalar(time):
            h = 0.5 / H2 * h
        else:
            h = 0.5 / np.expand_dims(np.asarray(H2, dtype=float), axis=-1) * h
    else:
        h = finite_line_source_approximation(
            time, alpha, dis, H1, D1, H2, D2, reaSource=reaSource,
            imgSource=imgSource, N=N)
    return h


def finite_line_source_equivalent_boreholes_vectorized(
        time, alpha, dis, wDis, H1, D1, H2, D2, N2, reaSource=True, imgSource=True):
    """
    Evaluate the equivalent Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Prieto and Cimmino
    [#eqFLSVec-PriCim2021]_. The equivalent FLS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2 H_2 N_{b,2}}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            \\sum_{G_1} \\sum_{G_2}
            e^{-d_{12}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            I_{real}(s) &= erfint((D_2-D_1+H_2)s) - erfint((D_2-D_1)s)

            &+ erfint((D_2-D_1-H_1)s) - erfint((D_2-D_1+H_2-H_1)s)

            I_{imag}(s) &= erfint((D_2+D_1+H_2)s) - erfint((D_2+D_1)s)

            &+ erfint((D_2+D_1+H_1)s) - erfint((D_2+D_1+H_2+H_1)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2 N_{b,2}}{H_1 N_{b,1}}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : array
        Unique radial distances to evaluate the FLS solution.
    wDis : array
        Number of instances of each unique radial distances.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    N2 : float or array,
        Number of segments represented by the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    This is a vectorized version of the :func:`finite_line_source` function
    using a fixed Gauss-Legendre quadrature over log-spaced panels to
    evaluate the integrals at all time values simultaneously. All arrays
    (dis, H1, D1, H2, D2) must follow numpy array broadcasting rules. If time
    is an array, the integrals for different time values are stacked on the
    last axis.


    References
    ----------
    .. [#eqFLSVec-PriCim2021] Prieto, C., & Cimmino, M.
       (2021). Thermal interactions in large irregular fields of geothermal
       boreholes: the method of equivalent borehole. Journal of Building
       Performance Simulation, 14 (4), 446-460.

    """
    # Integrand of the finite line source solution
    f = _finite_line_source_equivalent_boreholes_node_integrand(
        dis, wDis, H1, D1, H2, D2, N2, reaSource, imgSource)

    # Steady-state value of the integral (for infinite time values)
    def steady_state():
        ss = _finite_line_source_steady_state(
            np.asarray(dis, dtype=float).reshape(-1, 1),
            H1, D1, H2, D2, reaSource, imgSource)
        return np.asarray(wDis, dtype=float).T @ (2. * H2 * ss)

    # Evaluate integral for all time values
    d_min = float(np.min(dis)) if np.size(dis) > 0 else 1.
    h = _finite_line_source_integral_all_times(
        f, time, alpha, d_min, steady_state=steady_state)
    if _time_is_scalar(time):
        h = 0.5 / (N2*H2) * h
    else:
        h = 0.5 / np.expand_dims(
            np.asarray(N2*H2, dtype=float), axis=-1) * h
    return h


def finite_line_source_inclined_vectorized(
        time, alpha,
        rb1, x1, y1, H1, D1, tilt1, orientation1,
        x2, y2, H2, D2, tilt2, orientation2,
        reaSource=True, imgSource=True, M=11, approximation=False, N=10):
    """
    Evaluate the inclined Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the inclined FLS
    solution, as proposed by Lazzarotto [#incFLSVec-Lazzar2016]_.
    The inclined FLS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{H_1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            \\frac{1}{s}
            \\int_{0}^{1} (I_{real}(u, s)+I_{imag}(u, s)) du ds


            I_{real}(u, s) &=
            e^{-((x_1 - x_2)^2 + (y_1 - y_2)^2 + (D_1 - D_2)^2) s^2}

            &\\cdot (erf((u H_1 k_{0,real} + k_{2,real}) s)
             - erf((u H_1 k_{0,real} + k_{2,real} - H_2) s))

            &\\cdot  e^{(u^2 H_1^2 (k_{0,real}^2 - 1)
                + 2 u H_1 (k_{0,real} k_{2,real} - k_{1,real}) + k_{2,real}^2) s^2}
            du ds


            I_{imag}(u, s) &=
            -e^{-((x_1 - x_2)^2 + (y_1 - y_2)^2 + (D_1 + D_2)^2) s^2}

            &\\cdot (erf((u H_1 k_{0,imag} + k_{2,imag}) s)
             - erf((u H_1 k_{0,imag} + k_{2,imag} - H_2) s))

            &\\cdot e^{(u^2 H_1^2 (k_{0,imag}^2 - 1)
                + 2 u H_1 (k_{0,imag} k_{2,imag} - k_1) + k_{2,imag}^2) s^2}
            du ds


            k_{0,real} &=
            sin(\\beta_1) sin(\\beta_2) cos(\\theta_1 - \\theta_2)
            + cos(\\beta_1) cos(\\beta_2)


            k_{0,imag} &=
            sin(\\beta_1) sin(\\beta_2) cos(\\theta_1 - \\theta_2)
            - cos(\\beta_1) cos(\\beta_2)


            k_{1,real} &= sin(\\beta_1)
            (cos(\\theta_1) (x_1 - x_2) + sin(\\theta_1) (y_1 - y_2))
            + cos(\\beta_1) (D_1 - D_2)


            k_{1,imag} &= sin(\\beta_1)
            (cos(\\theta_1) (x_1 - x_2) + sin(\\theta_1) (y_1 - y_2))
            + cos(\\beta_1) (D_1 + D_2)


            k_{2,real} &= sin(\\beta_2)
            (cos(\\theta_2) (x_1 - x_2) + sin(\\theta_2) (y_1 - y_2))
            + cos(\\beta_2) (D_1 - D_2)


            k_{2,imag} &= sin(\\beta_2)
            (cos(\\theta_2) (x_1 - x_2) + sin(\\theta_2) (y_1 - y_2))
            - cos(\\beta_2) (D_1 + D_2)

    where :math:`\\beta_1` and :math:`\\beta_2` are the tilt angle of the
    boreholes (relative to vertical), and :math:`\\theta_1` and
    :math:`\\theta_2` are the orientation of the boreholes (relative to the
    x-axis).

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    rb1 : array
        Radii of the emitting heat sources.
    x1 : float or array
        x-Positions of the emitting heat sources.
    y1 : float or array
        y-Positions of the emitting heat sources.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    tilt1 : float or array
        Angles (in radians) from vertical of the emitting heat sources.
    orientation1 : float or array
        Directions (in radians) of the tilt the emitting heat sources.
    x2 : array
        x-Positions of the receiving heat sources.
    y2 : array
        y-Positions of the receiving heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    tilt2 : float or array
        Angles (in radians) from vertical of the receiving heat sources.
    orientation2 : float or array
        Directions (in radians) of the tilt the receiving heat sources.
    reaSource : bool, optional
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is true.
    M : int, optional
        Number of points for the Gauss-Legendre quadrature rule along the
        receiving heat sources.
        Default is 21.
    approximation : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#FLSVec-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Can be vector-valued.

    Notes
    -----
    This is a vectorized version of the :func:`finite_line_source` function
    using a fixed Gauss-Legendre quadrature over log-spaced panels to
    evaluate the integrals at all time values simultaneously. All arrays
    (x1, y1, H1, D1, tilt1, orientation1, x2, y2, H2, D2, tilt2,
    orientation2) must follow numpy array broadcasting rules.

    References
    ----------
    .. [#incFLSVec-Lazzar2016] Lazzarotto, A. (2016). A methodology for the
       calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 1, Renewable Energy, 86,
       1380-1393.

    """
    if not approximation:
        # Integrand of the inclined finite line source solution
        f, d_min = _finite_line_source_inclined_node_integrand(
            rb1, x1, y1, H1, D1, tilt1, orientation1,
            x2, y2, H2, D2, tilt2, orientation2,
            reaSource, imgSource, M)
        # Steady-state value of the integral (for infinite time values)
        steady_state = \
            lambda: 2. * H2 * _finite_line_source_inclined_steady_state(
                rb1, x1, y1, H1, D1, tilt1, orientation1,
                x2, y2, H2, D2, tilt2, orientation2,
                reaSource, imgSource, M=M)
        # Evaluate integral for all time values
        h = _finite_line_source_integral_all_times(
            f, time, alpha, d_min, steady_state=steady_state)
        if _time_is_scalar(time):
            h = 0.5 / H2 * h
        else:
            h = 0.5 / np.expand_dims(np.asarray(H2, dtype=float), axis=-1) * h
    else:
        h = finite_line_source_inclined_approximation(
            time, alpha, rb1, x1, y1, H1, D1, tilt1, orientation1,
            x2, y2, H2, D2, tilt2, orientation2,
            reaSource=reaSource, imgSource=imgSource, M=M, N=N)
    return h


@lru_cache(maxsize=None)
def _roots_legendre_cached(deg):
    """Cached nodes and weights of Gauss-Legendre quadrature."""
    x, w = roots_legendre(deg)
    return x, w


def _finite_line_source_coefficients(H1, D1, H2, D2, reaSource, imgSource):
    """
    Coefficients (p, q) of the terms of the integrand of the finite line
    source solution, such that the integrand is given by:

        f(s) = s**-2 * exp(-dis**2 * s**2) * sum_j (p_j * erfint(q_j * s))

    Parameters
    ----------
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    p : array
        Signs of the terms of the integrand, shape (nq,).
    q : array
        Coefficients of the terms of the integrand. The last axis is of
        length nq.

    """
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        p = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1,
                      D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1],
                     axis=-1)
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
    else:
        # No heat source
        p = np.zeros(1)
        q = np.zeros(
            np.broadcast_shapes(
                *[np.shape(arg) for arg in (H1, D1, H2, D2)]) + (1,))
    return p, q


def _time_is_scalar(time):
    """Return True if the time value is a scalar."""
    return isinstance(time, (np.floating, float, np.integer, int))


def _finite_line_source_integral_all_times(
        f, time, alpha, d_min, steady_state=None, deg=10, log_ratio=0.5):
    """
    Evaluate the integral of the finite line source solution at all times.

    The cumulative integrals:

        I_k = int_{a_k}^{inf} f(s) ds,    a_k = 1 / sqrt(4 * alpha * t_k)

    are evaluated for all time values simultaneously using composite
    Gauss-Legendre quadrature over log-spaced panels between consecutive
    integration bounds. The integral is truncated at
    s_max = 9 / d_min, where the integrand is negligible (the integrand
    decays at least as fast as exp(-d_min**2 * s**2) / s, so that the
    neglected part of the integral is below 1e-36 relative to the scale of
    the integrand coefficients).

    Parameters
    ----------
    f : callable
        Integrand of the finite line source solution. Accepts an array of
        integration points s of shape (ns,) and returns an array of shape
        (..., ns).
    time : float or array, shape (nt,)
        Value of time (in seconds) for which the integral is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    d_min : float
        Minimum radial distance (in meters) in the integrand. This provides
        the decay scale used to truncate the integral.
    steady_state : callable or None, optional
        Callable that returns the steady-state value of the integral
        (i.e. the integral from 0 to infinity), of shape (...). Only used
        for infinite time values.
        Default is None.
    deg : int, optional
        Degree of the Gauss-Legendre quadrature over each panel.
        Default is 10.
    log_ratio : float, optional
        Maximum width of a panel in logarithmic space,
        i.e. log(s_upper) - log(s_lower) <= log_ratio.
        Default is 0.5.

    Returns
    -------
    I : array
        Values of the integral, shape (..., nt), or (...,) if time is a
        float.

    """
    scalar_time = _time_is_scalar(time)
    t = np.atleast_1d(np.asarray(time, dtype=float))
    nt = t.size
    # Shape of the integrand
    out_shape = np.shape(f(np.ones(1)))[:-1]
    out_size = int(np.prod(out_shape, dtype=int))
    if out_size == 0:
        # Empty integrand
        if scalar_time:
            return np.zeros(out_shape)
        return np.zeros(out_shape + (nt,))
    # Steady-state solution (i.e. the full integral) for infinite time
    # values
    is_inf = np.isinf(t)
    if np.any(is_inf):
        if steady_state is None:
            raise ValueError(
                'A steady-state evaluator is required for time = np.inf.')
        integrals = np.empty(out_shape + (nt,))
        integrals[..., is_inf] = np.expand_dims(
            np.broadcast_to(steady_state(), out_shape), axis=-1)
        if not np.all(is_inf):
            integrals[..., ~is_inf] = _finite_line_source_integral_all_times(
                f, t[~is_inf], alpha, d_min, deg=deg, log_ratio=log_ratio)
        if scalar_time:
            integrals = integrals[..., 0]
        return integrals
    # Integration bounds in ascending order (i.e. descending time). The
    # bound is infinite for time t=0 : the corresponding integral is zero
    # and its pieces are clipped to zero width at s_max.
    i_sort = np.argsort(t)
    with np.errstate(divide='ignore'):
        bounds = 1.0 / np.sqrt(4.0 * alpha * t[i_sort][::-1])
    # Truncation of the integral at s_max, where the integrand is negligible
    s_max = 9.0 / d_min
    if np.isfinite(bounds[-1]):
        s_max = max(s_max, bounds[-1])
    elif np.any(np.isfinite(bounds)):
        s_max = max(s_max, np.max(bounds[np.isfinite(bounds)]))
    # Pieces of the integral between consecutive integration bounds
    lo = np.minimum(bounds, s_max)
    hi = np.append(lo[1:], s_max)
    # Number of log-spaced panels per piece
    log_width = np.log(hi) - np.log(lo)
    nPanels = np.ceil(log_width / log_ratio - 1e-12).astype(int)
    # Log-spaced panel edges within each piece
    panel_edges = [
        np.append(lo_i * np.exp(np.arange(n_i) * (w_i / n_i)), hi_i)
        if n_i > 0 else np.empty(0)
        for lo_i, hi_i, w_i, n_i in zip(lo, hi, log_width, nPanels)]
    panel_lo = np.concatenate(
        [edges[:-1] if len(edges) > 0 else edges for edges in panel_edges])
    panel_hi = np.concatenate(
        [edges[1:] if len(edges) > 0 else edges for edges in panel_edges])
    nPanelsTotal = len(panel_lo)
    # Gauss-Legendre nodes and weights on each panel
    x, w = _roots_legendre_cached(deg)
    panel_mid = 0.5 * (panel_lo + panel_hi)
    panel_half = 0.5 * (panel_hi - panel_lo)
    s_nodes = (panel_mid[:, np.newaxis]
               + panel_half[:, np.newaxis] * x).flatten()
    w_nodes = (panel_half[:, np.newaxis] * w).flatten()
    # Integrals over each panel, evaluated in chunks to limit memory use
    panel_integrals = np.empty(out_shape + (nPanelsTotal,))
    chunk_size = max(2**23 // max(deg * out_size, 1), 1)
    for i0 in range(0, nPanelsTotal, chunk_size):
        i1 = min(i0 + chunk_size, nPanelsTotal)
        values = f(s_nodes[i0*deg:i1*deg]) * w_nodes[i0*deg:i1*deg]
        panel_integrals[..., i0:i1] = values.reshape(
            out_shape + (i1 - i0, deg)).sum(axis=-1)
    # Integrals over each piece
    panel_cumsum = np.concatenate(
        (np.zeros(out_shape + (1,)),
         np.cumsum(panel_integrals, axis=-1)),
        axis=-1)
    piece_ends = np.cumsum(nPanels)
    piece_starts = piece_ends - nPanels
    piece_integrals = (
        panel_cumsum[..., piece_ends] - panel_cumsum[..., piece_starts])
    # Cumulative integrals I_k (in descending order of integration bounds,
    # i.e. ascending order of time)
    I = np.cumsum(piece_integrals[..., ::-1], axis=-1)
    # Return the integrals in the original order of the time values
    integrals = np.empty(out_shape + (nt,))
    integrals[..., i_sort] = I
    if scalar_time:
        integrals = integrals[..., 0]
    return integrals


def _finite_line_source_laplace(
        p, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True):
    """
    Evaluate the Laplace-domain transfer function of the Finite Line Source
    (FLS) solution.

    The FLS solution is expressed by an integral over the variable s with a
    sharp cutoff in time:

        h(t) = int_0^inf f(s) * u(t - 1 / (4 * alpha * s**2)) ds

    where u is the unit step function. The Laplace-domain transfer function
    (i.e. p times the Laplace transform of the step response h) follows by
    transforming the step function:

        p * L{h}(p) = int_0^inf f(s) * exp(-p / (4 * alpha * s**2)) ds

    i.e. the same integrand f(s) weighted by a smooth exponential factor.
    At p = 0, the transfer function equals the steady-state FLS solution.

    Parameters
    ----------
    p : float or array, shape (nP,)
        Laplace parameter (in 1/seconds). Must be real and non-negative.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.

    Returns
    -------
    h : array
        Values of the Laplace-domain transfer function of the FLS solution,
        shape (..., nP), or (...,) if p is a float. The transfer function is
        dimensionless and normalized as the thermal response factors, i.e.
        the Laplace transform of the borehole wall temperature variation is:

        .. math::
            \\Delta \\hat{T}_{b,2}(p) =
            \\frac{1}{2 \\pi k_s H_2} (p \\hat{h}(p)) \\hat{Q}_1(p)

    """
    scalar_p = _time_is_scalar(p)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    # Integrand of the finite line source solution
    f = _finite_line_source_node_integrand(
        dis, H1, D1, H2, D2, reaSource, imgSource)
    d_min = float(np.min(dis)) if np.size(dis) > 0 else 1.
    h = np.stack(
        [_finite_line_source_laplace_integral(f, p_m, alpha, d_min)
         if p_m > 0.
         else 2. * H2 * _finite_line_source_steady_state(
             dis, H1, D1, H2, D2, reaSource, imgSource)
         for p_m in p],
        axis=-1)
    if scalar_p:
        h = 0.5 / H2 * h[..., 0]
    else:
        h = 0.5 / np.expand_dims(np.asarray(H2, dtype=float), axis=-1) * h
    return h


def _finite_line_source_laplace_integral(
        f, p, alpha, d_min, deg=10, log_ratio=0.5):
    """
    Evaluate the integral of the finite line source integrand weighted by
    the Laplace-domain kernel:

        I(p) = int_0^inf f(s) * exp(-p / (4 * alpha * s**2)) ds

    using composite Gauss-Legendre quadrature over log-spaced panels. The
    integral is truncated at s_max = 9 / d_min (where the integrand is
    negligible) and below s_lo (where the weighting factor is below
    ~1e-15). Panels are refined near s_lo, where the weighting factor
    varies rapidly.

    Parameters
    ----------
    f : callable
        Integrand of the finite line source solution. Accepts an array of
        integration points s of shape (ns,) and returns an array of shape
        (..., ns).
    p : float
        Laplace parameter (in 1/seconds). Must be positive.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    d_min : float
        Minimum radial distance (in meters) in the integrand.
    deg : int, optional
        Degree of the Gauss-Legendre quadrature over each panel.
        Default is 10.
    log_ratio : float, optional
        Maximum width of a panel in logarithmic space.
        Default is 0.5.

    Returns
    -------
    I : array
        Values of the integral, shape (...,).

    """
    # Shape of the integrand
    out_shape = np.shape(f(np.ones(1)))[:-1]
    out_size = int(np.prod(out_shape, dtype=int))
    # Cutoff of the weighting factor exp(-E) at E_cut (exp(-34.5) ~ 1e-15)
    E_cut = 34.5
    s_max = 9.0 / d_min
    s_lo = np.sqrt(p / (4. * alpha * E_cut))
    if out_size == 0 or s_lo >= s_max:
        return np.zeros(out_shape)
    # Log-spaced panel edges, refined where the weighting factor varies
    # rapidly : the local logarithmic derivative of the weighting factor is
    # 2 * E(s), with E(s) = p / (4 * alpha * s**2). Panel widths are
    # limited to 4 / E so that the Gauss-Legendre rule resolves the
    # variation of the weighting factor.
    x_lo = np.log(s_lo)
    x_hi = np.log(s_max)
    edges = [x_lo]
    while edges[-1] < x_hi:
        E = p / (4. * alpha * np.exp(2. * edges[-1]))
        width = min(log_ratio, 4. / max(E, 1e-12))
        edges.append(min(edges[-1] + width, x_hi))
    s_edges = np.exp(np.asarray(edges))
    panel_lo = s_edges[:-1]
    panel_hi = s_edges[1:]
    nPanels = len(panel_lo)
    # Gauss-Legendre nodes and weights on each panel
    x, w = _roots_legendre_cached(deg)
    panel_mid = 0.5 * (panel_lo + panel_hi)
    panel_half = 0.5 * (panel_hi - panel_lo)
    s_nodes = (panel_mid[:, np.newaxis]
               + panel_half[:, np.newaxis] * x).flatten()
    w_nodes = (panel_half[:, np.newaxis] * w).flatten()
    # Weighted integral, evaluated in chunks to limit memory use
    weights = w_nodes * np.exp(-p / (4. * alpha * np.square(s_nodes)))
    I = np.zeros(out_shape)
    chunk_size = max(2**23 // max(deg * out_size, 1), 1) * deg
    for i0 in range(0, nPanels * deg, chunk_size):
        i1 = min(i0 + chunk_size, nPanels * deg)
        I += f(s_nodes[i0:i1]) @ weights[i0:i1]
    return I


def _finite_line_source_node_integrand(dis, H1, D1, H2, D2, reaSource, imgSource):
    """
    Integrand of the finite line source solution, evaluated at an array of
    integration points.

    Parameters
    ----------
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Accepts an array of
        integration points s of shape (ns,) and returns an array of shape
        (..., ns), where (...) is the broadcast shape of the input
        parameters.

    """
    p, q = _finite_line_source_coefficients(
        H1, D1, H2, D2, reaSource, imgSource)
    G = _erfint_linear_combination(p, q)
    E = _exp_outer_deduplicated(np.square(dis))

    def f(s):
        return E(s) * G(s) / np.square(s)

    return f


def _erfint_linear_combination(p, q):
    """
    Efficient evaluation of the linear combination of integrals of the
    error function in the integrand of the finite line source solution.

    The returned callable evaluates:

        G(s) = sum_i (p_i * erfint(q_i * s))

    at an array of integration points s. Since erfint is an even function,
    the (typically few) unique absolute values of the coefficients q are
    identified so that erfint is only evaluated once per unique value. The
    linear combination is then applied through a sparse matrix-matrix
    product.

    Parameters
    ----------
    p : array
        Signs of the terms of the integrand, shape (nq,).
    q : array
        Coefficients of the terms of the integrand. The last axis is of
        length nq.

    Returns
    -------
    G : callable
        Accepts an array of integration points s of shape (ns,) and returns
        an array of shape q.shape[:-1] + (ns,).

    """
    from scipy.sparse import csr_matrix
    out_shape = q.shape[:-1]
    nq = q.shape[-1]
    nElements = int(np.prod(out_shape, dtype=int))
    # Unique absolute values of the coefficients q (erfint is even)
    q_unique, inverse = np.unique(np.abs(q).flatten(), return_inverse=True)
    # Sparse matrix of the linear combination, such that
    # G = combination @ erfint(outer(q_unique, s))
    combination = csr_matrix(
        (np.tile(p, nElements),
         (np.repeat(np.arange(nElements), nq), inverse)),
        shape=(nElements, len(q_unique)))

    def G(s):
        return (combination @ erfint(np.multiply.outer(q_unique, s))).reshape(
            out_shape + np.shape(s))

    return G


def _exp_outer_deduplicated(dd):
    """
    Efficient evaluation of the exponential of the outer product of squared
    distances and squared integration points.

    The returned callable evaluates:

        E(s) = exp(-outer(dd, s**2))

    at an array of integration points s. The exponential is only evaluated
    once per unique value of dd.

    Parameters
    ----------
    dd : float or array
        Squared radial distances.

    Returns
    -------
    E : callable
        Accepts an array of integration points s of shape (ns,) and returns
        an array of shape np.shape(dd) + (ns,).

    """
    dd_shape = np.shape(dd)
    dd_unique, inverse = np.unique(
        np.asarray(dd, dtype=float).flatten(), return_inverse=True)
    if len(dd_unique) == np.prod(dd_shape, dtype=int):
        # No duplicated distances
        def E(s):
            return np.exp(-np.multiply.outer(dd, np.square(s)))
    else:
        def E(s):
            return np.exp(
                -np.multiply.outer(dd_unique, np.square(s)))[inverse].reshape(
                    dd_shape + np.shape(s))

    return E


def _finite_line_source_equivalent_boreholes_node_integrand(
        dis, wDis, H1, D1, H2, D2, N2, reaSource, imgSource):
    """
    Integrand of the equivalent finite line source solution, evaluated at an
    array of integration points.

    Parameters
    ----------
    dis : array
        Unique radial distances to evaluate the FLS solution.
    wDis : array
        Number of instances of each unique radial distances.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    N2 : float or array,
        Number of segments represented by the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Accepts an array of
        integration points s of shape (ns,) and returns an array of shape
        (..., ns).

    """
    p, q = _finite_line_source_coefficients(
        H1, D1, H2, D2, reaSource, imgSource)
    G = _erfint_linear_combination(p, q)
    dd = np.square(np.asarray(dis, dtype=float).flatten())
    wDisT = np.asarray(wDis, dtype=float).T

    def f(s):
        E = wDisT @ np.exp(-np.multiply.outer(dd, np.square(s)))
        return np.expand_dims(E, axis=-2) * G(s) / np.square(s)

    return f


def _finite_line_source_inclined_node_integrand(
        rb1, x1, y1, H1, D1, tilt1, orientation1,
        x2, y2, H2, D2, tilt2, orientation2, reaSource, imgSource, M):
    """
    Integrand of the inclined finite line source solution, evaluated at an
    array of integration points.

    Parameters
    ----------
    rb1 : array
        Radii of the emitting heat sources.
    x1, y1 : float or array
        Positions of the emitting heat sources.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    tilt1 : float or array
        Angles (in radians) from vertical of the emitting heat sources.
    orientation1 : float or array
        Directions (in radians) of the tilt the emitting heat sources.
    x2, y2 : float or array
        Positions of the receiving heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    tilt2 : float or array
        Angles (in radians) from vertical of the receiving heat sources.
    orientation2 : float or array
        Directions (in radians) of the tilt the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
    M : int
        Number of points for the Gauss-Legendre quadrature rule along the
        receiving heat sources.

    Returns
    -------
    f : callable
        Integrand of the inclined finite line source solution. Accepts an
        array of integration points s of shape (ns,) and returns an array
        of shape (..., ns), where (...) is the broadcast shape of the input
        parameters.
    d_min : float
        Decay scale (in meters) of the integrand, such that the integrand
        decays at least as fast as exp(-d_min**2 * s**2).

    """
    output_shape = np.broadcast_shapes(
            *[np.shape(arg) for arg in (
                rb1, x1, y1, H1, D1, tilt1, orientation1,
                x2, y2, H2, D2, tilt2, orientation2)])
    output_ndim = len(output_shape)
    # Expand parameters with a trailing axis for the integration points s
    expand = lambda arr: np.reshape(
        np.broadcast_to(arr, output_shape), output_shape + (1,))
    # Roots for Gauss-Legendre quadrature, with an additional trailing axis
    # for the integration points s
    x, w = _roots_legendre_cached(M)
    u = (0.5 * x + 0.5).reshape((-1,) + (1,) * (output_ndim + 1))
    w = w / 2
    # Sines and cosines of tilt (b: beta) and orientation (t: theta)
    sb1 = np.sin(tilt1)
    sb2 = np.sin(tilt2)
    cb1 = np.cos(tilt1)
    cb2 = np.cos(tilt2)
    dx = x1 - x2
    dy = y1 - y2
    rr = np.maximum(dx**2 + dy**2, rb1**2)
    H1e = expand(H1)
    H2e = expand(H2)
    # Terms of the integrand for the real and image sources. Each term is a
    # tuple (sign, c, k), where the term of the integrand is given by:
    #    sign * exp(-c * s**2) * (erf(k * s) - erf((k - H2) * s))
    # with c and k evaluated at the M quadrature points u along the emitting
    # heat source.
    terms = []
    if reaSource:
        dzRea = D1 - D2
        kRea_0 = sb1 * sb2 * np.cos(orientation1 - orientation2) + cb1 * cb2
        kRea_1 = sb1 * (np.cos(orientation1) * dx + np.sin(orientation1) * dy) + cb1 * dzRea
        kRea_2 = sb2 * (np.cos(orientation2) * dx + np.sin(orientation2) * dy) + cb2 * dzRea
        cRea = (expand(rr + dzRea**2)
                - (u**2 * H1e**2 * (expand(kRea_0)**2 - 1)
                   + 2 * u * H1e * expand(kRea_0 * kRea_2 - kRea_1)
                   + expand(kRea_2)**2))
        kRea = u * H1e * expand(kRea_0) + expand(kRea_2)
        terms.append((1., cRea, kRea))
    if imgSource:
        dzImg = D1 + D2
        kImg_0 = sb1 * sb2 * np.cos(orientation1 - orientation2) - cb1 * cb2
        kImg_1 = sb1 * (np.cos(orientation1) * dx + np.sin(orientation1) * dy) + cb1 * dzImg
        kImg_2 = sb2 * (np.cos(orientation2) * dx + np.sin(orientation2) * dy) - cb2 * dzImg
        # The radial distance is clamped at the borehole radius when the
        # real part of the FLS solution is included
        rrImg = rr if reaSource else dx**2 + dy**2
        cImg = (expand(rrImg + dzImg**2)
                - (u**2 * H1e**2 * (expand(kImg_0)**2 - 1)
                   + 2 * u * H1e * expand(kImg_0 * kImg_2 - kImg_1)
                   + expand(kImg_2)**2))
        kImg = u * H1e * expand(kImg_0) + expand(kImg_2)
        terms.append((-1., cImg, kImg))
    if len(terms) == 0:
        d_min = 1.
        f = lambda s: np.zeros(output_shape + np.shape(s))
        return f, d_min
    # Decay scale of the integrand, clamped to a fraction of the borehole
    # radius to protect against (nearly) intersecting boreholes
    if int(np.prod(output_shape, dtype=int)) > 0:
        c_min = np.min([np.min(c) for (_, c, _) in terms])
        d_min = np.sqrt(max(c_min, np.min(np.square(rb1)) * 1e-4))
    else:
        d_min = 1.

    def f(s):
        total = np.zeros((M,) + output_shape + np.shape(s))
        for (sign, c, k) in terms:
            total += sign * np.exp(-c * np.square(s)) \
                * (erf(k * s) - erf((k - H2e) * s))
        return np.einsum('i,i...->...', w, total) * (H1e / s)

    return f, d_min

def _finite_line_source_steady_state(dis, H1, D1, H2, D2, reaSource, imgSource):
    """
    Steady-state finite line source solution.

    Parameters
    ----------
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    h : Steady-state finite line source solution.

    Notes
    -----
    All arrays (dis, H1, D1, H2, D2) must follow numpy array broadcasting
    rules.

    """
    # Steady-state solution
    if reaSource or imgSource:
        p, q = _finite_line_source_coefficients(
            H1, D1, H2, D2, reaSource, imgSource)
        dis = np.expand_dims(dis, axis=-1)
        qpd = np.sqrt(q**2 + dis**2)
        h = 0.5 / H2 * np.inner(p, q * np.log(q + qpd) - qpd)
    else:
        # No heat source
        h = np.zeros(np.broadcast_shapes(
            *[np.shape(arg) for arg in (dis, H1, D1, H2, D2)]))
    return h


def _finite_line_source_inclined_steady_state(
        rb1, x1, y1, H1, D1, tilt1, orientation1, x2, y2, H2, D2, tilt2,
        orientation2, reaSource, imgSource, M=11):
    """
    Steady-state inclined Finite Line Source (FLS) solution.

    Parameters
    ----------
    rb1 : array
        Radii of the emitting heat sources.
    x1 : float or array
        x-Positions of the emitting heat sources.
    y1 : float or array
        y-Positions of the emitting heat sources.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    tilt1 : float or array
        Angles (in radians) from vertical of the emitting heat sources.
    orientation1 : float or array
        Directions (in radians) of the tilt the emitting heat sources.
    x2 : array
        x-Positions of the receiving heat sources.
    y2 : array
        y-Positions of the receiving heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    tilt2 : float or array
        Angles (in radians) from vertical of the receiving heat sources.
    orientation2 : float or array
        Directions (in radians) of the tilt the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
    M : int
        Number of points for the Gauss-Legendre quadrature rule along the
        receiving heat sources.
        Default is 11.

    Returns
    -------
    h : Steady-state inclined finite line source solution.

    Notes
    -----
    All arrays (dis, H1, D1, H2, D2) must follow numpy array broadcasting
    rules.

    """
    output_shape = np.broadcast_shapes(
            *[np.shape(arg) for arg in (
                rb1, x1, y1, H1, D1, tilt1, orientation1,
                x2, y2, H2, D2, tilt2, orientation2)])
    # Roots
    x, w = _roots_legendre_cached(M)
    u = (0.5 * x + 0.5).reshape((-1,) + (1,) * len(output_shape))
    w = w / 2
    # Params
    sb1 = np.sin(tilt1)
    sb2 = np.sin(tilt2)
    cb1 = np.cos(tilt1)
    cb2 = np.cos(tilt2)
    dx = x1 - x2
    dy = y1 - y2
    # Steady-state solution
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        dzRea = D1 - D2
        dzImg = D1 + D2
        rr = np.maximum(dx**2 + dy**2, rb1**2)
        kRea_0 = sb1 * sb2 * np.cos(orientation1 - orientation2) + cb1 * cb2
        kImg_0 = sb1 * sb2 * np.cos(orientation1 - orientation2) - cb1 * cb2
        kRea_1 = sb1 * (np.cos(orientation1) * dx + np.sin(orientation1) * dy) + cb1 * dzRea
        kImg_1 = sb1 * (np.cos(orientation1) * dx + np.sin(orientation1) * dy) + cb1 * dzImg
        kRea_2 = sb2 * (np.cos(orientation2) * dx + np.sin(orientation2) * dy) + cb2 * dzRea
        kImg_2 = sb2 * (np.cos(orientation2) * dx + np.sin(orientation2) * dy) - cb2 * dzImg
        h = 0.5 * H1 / H2 * ((
            np.log((2 * H2 * np.sqrt(rr + dzRea**2 + u**2*H1**2 + H2**2 + 2*u*H1*kRea_1 - 2*H2*kRea_2 - 2*u*H1*H2*kRea_0) + 2*H2**2 - 2*H2*kRea_2 - 2*u*H1*H2*kRea_0) / ((2 * H2 * np.sqrt(rr + dzRea**2 + u**2*H1**2 + 2*u*H1*kRea_1) - 2*H2*kRea_2 - 2*u*H1*H2*kRea_0)))
            - np.log((2 * H2 * np.sqrt(dx**2 + dy**2 + dzImg**2 + u**2*H1**2 + H2**2 + 2*u*H1*kImg_1 - 2*H2*kImg_2 - 2*u*H1*H2*kImg_0) + 2*H2**2 - 2*H2*kImg_2 - 2*u*H1*H2*kImg_0) / ((2 * H2 * np.sqrt(dx**2 + dy**2 + dzImg**2 + u**2*H1**2 + 2*u*H1*kImg_1) - 2*H2*kImg_2 - 2*u*H1*H2*kImg_0)))
            ).T @ w).T
    elif reaSource:
        # Real FLS solution
        dzRea = D1 - D2
        rr = np.maximum(dx**2 + dy**2, rb1**2)
        kRea_0 = sb1 * sb2 * np.cos(orientation1 - orientation2) + cb1 * cb2
        kRea_1 = sb1 * (np.cos(orientation1) * dx + np.sin(orientation1) * dy) + cb1 * dzRea
        kRea_2 = sb2 * (np.cos(orientation2) * dx + np.sin(orientation2) * dy) + cb2 * dzRea
        h = 0.5 * H1 / H2 * ((
            np.log((2 * H2 * np.sqrt(rr + dzRea**2 + u**2*H1**2 + H2**2 + 2*u*H1*kRea_1 - 2*H2*kRea_2 - 2*u*H1*H2*kRea_0) + 2*H2**2 - 2*H2*kRea_2 - 2*u*H1*H2*kRea_0) / ((2 * H2 * np.sqrt(rr + dzRea**2 + u**2*H1**2 + 2*u*H1*kRea_1) - 2*H2*kRea_2 - 2*u*H1*H2*kRea_0)))
            ).T @ w).T
    elif imgSource:
        # Image FLS solution
        dzImg = D1 + D2
        kImg_0 = sb1 * sb2 * np.cos(orientation1 - orientation2) - cb1 * cb2
        kImg_1 = sb1 * (np.cos(orientation1) * dx + np.sin(orientation1) * dy) + cb1 * dzImg
        kImg_2 = sb2 * (np.cos(orientation2) * dx + np.sin(orientation2) * dy) - cb2 * dzImg
        h = 0.5 * H1 / H2 * ((
            - np.log((2 * H2 * np.sqrt(dx**2 + dy**2 + dzImg**2 + u**2*H1**2 + H2**2 + 2*u*H1*kImg_1 - 2*H2*kImg_2 - 2*u*H1*H2*kImg_0) + 2*H2**2 - 2*H2*kImg_2 - 2*u*H1*H2*kImg_0) / ((2 * H2 * np.sqrt(dx**2 + dy**2 + dzImg**2 + u**2*H1**2 + 2*u*H1*kImg_1) - 2*H2*kImg_2 - 2*u*H1*H2*kImg_0)))
            ).T @ w).T
    else:
        # No heat source
        h = np.zeros(output_shape)
    return h
