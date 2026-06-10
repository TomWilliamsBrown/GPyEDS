# GPyEDS
[![codecov](https://codecov.io/gh/norberttoth398/GPyEDS/graph/badge.svg?token=I4Ukc39QBJ)](https://codecov.io/gh/norberttoth398/GPyEDS) [![Documentation Status](https://readthedocs.org/projects/gpyeds/badge/?version=latest)](https://gpyeds.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13837097.svg)](https://doi.org/10.5281/zenodo.13837097)

Autoencoder-based unsupervised feature-extraction toolbox for energy-dispersive
spectroscopy (EDS) data.

## Installation

GPyEDS requires **Python 3.10-3.12** (3.12 recommended). The Gaussian-process and
neural-network models are built on `gpflux`, which caps the stack at TensorFlow 2.16
/ Python 3.12.

### From a local clone

    cd GPyEDS
    pip install .            # core toolbox
    pip install '.[tf]'      # core + GP/NN models (TensorFlow, GPflow, GPflux)

For a reproducible environment from the pinned `uv.lock` (needs [uv](https://docs.astral.sh/uv/)):

    uv sync --extra tf --extra test

### Build a redistributable wheel (no Git host required)

GPyEDS builds a single platform-independent wheel you can copy to any machine:

    uv build                 # or:  pip install build && python -m build
    # produces dist/gpyeds-0.0.1-py3-none-any.whl  (+ .tar.gz source dist)

Copy the `.whl` to the target machine and install it directly; the dependencies are
resolved from PyPI:

    pip install gpyeds-0.0.1-py3-none-any.whl            # core
    pip install 'gpyeds-0.0.1-py3-none-any.whl[tf]'      # core + GP/NN models

## Usage

    import GPyEDS
    from GPyEDS import mean_centre

The GP and neural-network models (`GPyEDS.GPAM`, `GPyEDS.nn`) use the Keras-2 API.
TensorFlow >= 2.16 defaults to Keras 3, so set the following before importing them
(the core modules need nothing extra):

    export TF_USE_LEGACY_KERAS=1
