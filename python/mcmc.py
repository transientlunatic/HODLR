#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import kplr
import fitsio
import numpy as np
import matplotlib.pyplot as pl
from transit import ldlc_simple

import hodlr

# Fixed parameters.
texp, tol, maxdepth = kplr.EXPOSURE_TIMES[0] / 86400., 0.1, 2
q1, q2 = 0.4, 0.3
period, t0, tau, ror, b = 100., 10., 0.5, 0.02, 0.5


def model(fstar, q1, q2, t0, tau, ror, b):
    u1, u2 = 2*q1*q2, q1*(1-2*q2)
    lc = ldlc_simple(t, u1, u2, period, t0, tau, ror, b, texp, tol, maxdepth)
    return fstar * lc


def lnlike(p):
    lna, lns, fstar, q1, q2, t0, tau, ror, b = p
    a2, s2 = np.exp(2*lna), np.exp(2*lns)

    # Compute the model and residuals.
    res = f - model(fstar, q1, q2, t0, tau, ror, b)

    pl.clf()
    pl.plot(t, res, ".k")
    pl.savefig("res.png")

    # Solve the GP likelihood.
    matrix = hodlr.HODLR(a2, s2, t, fe2)
    alpha = matrix.solve(res)
    logdet = matrix.logdet()
    if not np.isfinite(logdet):
        return -np.inf
    return -0.5 * (np.dot(res, alpha) + logdet + norm)


# Load the data.
data = fitsio.read("data/kplr010593626-2011024051157_slc.fits")
t, f, fe, q = (data["TIME"], data["SAP_FLUX"], data["SAP_FLUX_ERR"],
               data["SAP_QUALITY"])

# Mask missing data.
m = np.isfinite(f) * np.isfinite(t) * (q == 0)
t, f, fe, q = t[m], f[m], fe[m], q[m]

# Normalize by the median uncertainty for numerical stability.
f, fe = f / np.median(fe), fe / np.median(fe)
fe2 = fe * fe

# Normalize the times.
t -= np.min(t)

# Pre-compute the normalization factor for the log-likelihood.
norm = len(t) * np.log(2 * np.pi)

# Inject a transit.
p0 = np.array([1e-10, 1.0, np.median(f), q1, q2, t0, tau, ror, b])
f *= model(1, *(p0[3:]))

# Compute the log likelihood.
import time
strt = time.time()
print(lnlike(p0))
print(time.time() - strt)

# Plot the initial data.
pl.clf()
pl.plot(t, f, ".k", alpha=0.3)
pl.savefig("data.png")
