"""Deprecated compatibility shim for the old top-level ``MCMC_emcee`` module.

The contents of ``MCMC_emcee`` were folded into :mod:`GPyEDS.EDS2CHEM.mcmc`
during the legacy cleanup. This module exists only so that pre-migration
scripts and notebooks that do ``from MCMC_emcee import align`` keep working.

New code should import from :mod:`GPyEDS.EDS2CHEM.mcmc` directly.

The one behavioural difference between the old and new APIs is the signature
of :func:`align`:

* legacy:  ``align(inputmatrix, n, theta)`` -- ``n`` is the number of points
* current: ``align(inputmatrix, theta, pos)`` -- ``pos`` is a positions array

This shim preserves the *legacy* call signature.
"""

import warnings

import numpy as np

# Re-export the canonical public API unchanged. Only ``align`` had its
# signature changed, so it is overridden below.
from GPyEDS.EDS2CHEM.mcmc import (  # noqa: F401
    draw_line,
    draw_proj_box,
    align_once,
    logfuncs,
    simple_logfuncs,
    MCMC_run,
    Simple_MCMC_run,
)
from GPyEDS.EDS2CHEM.mcmc import align as _align_current

__all__ = [
    "draw_line",
    "draw_proj_box",
    "align",
    "align_once",
    "logfuncs",
    "simple_logfuncs",
    "MCMC_run",
    "Simple_MCMC_run",
]

warnings.warn(
    "The top-level 'MCMC_emcee' module is deprecated; its contents now live in "
    "'GPyEDS.EDS2CHEM.mcmc'. Update imports to "
    "'from GPyEDS.EDS2CHEM.mcmc import ...'.",
    DeprecationWarning,
    stacklevel=2,
)


def align(inputmatrix, n, theta, **kwargs):
    """Legacy ``align`` wrapper.

    Preserves the old ``align(inputmatrix, n, theta)`` calling convention,
    where ``n`` is the number of transect points. It maps onto the current
    :func:`GPyEDS.EDS2CHEM.mcmc.align`, which instead takes an explicit
    positions array as its final argument.

    Args:
        inputmatrix: 2-D image the transect is sampled from.
        n: Number of transect points (scalar). For convenience an explicit
            positions array is also accepted and passed straight through.
        theta: Parameter vector ``[ax, ay, bx, by, ww, m, b, phi]``.

    Returns:
        Modelled profile of length ``n`` (``slope * value + intercept``).
    """
    warnings.warn(
        "MCMC_emcee.align(inputmatrix, n, theta) is deprecated. Use "
        "GPyEDS.EDS2CHEM.mcmc.align(inputmatrix, theta, pos) with an explicit "
        "positions array.",
        DeprecationWarning,
        stacklevel=2,
    )
    pos = np.asarray(n)
    if pos.ndim == 0:  # scalar count -> build the 0..n-1 positions array
        pos = np.arange(int(pos))
    return _align_current(inputmatrix, theta, pos, **kwargs)
