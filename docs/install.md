# Installation

GPyEDS requires **Python 3.10-3.12** (3.12 recommended). The Gaussian-process and
neural-network models are built on `gpflux`, which caps the stack at TensorFlow 2.16
/ Python 3.12.

## Install directly from GitHub (recommended)

No clone or build artifacts needed:

    pip install "git+https://github.com/TomWilliamsBrown/GPyEDS.git"

With the optional GP / neural-network models (TensorFlow, GPflow, GPflux):

    pip install "GPyEDS[tf] @ git+https://github.com/TomWilliamsBrown/GPyEDS.git"

Pin to a tagged release for reproducible installs:

    pip install "GPyEDS[tf] @ git+https://github.com/TomWilliamsBrown/GPyEDS.git@1.0.1"

`uv` users can do the same:

    uv pip install "git+https://github.com/TomWilliamsBrown/GPyEDS.git"
    # or add it to a project:
    uv add "GPyEDS @ git+https://github.com/TomWilliamsBrown/GPyEDS.git"

The bare URL installs the repository's **default branch**; append `@<branch-or-tag>`
to target a specific ref.

## From a clone (for development)

    git clone https://github.com/TomWilliamsBrown/GPyEDS.git
    cd GPyEDS
    pip install -e '.[tf]'           # editable install incl. the GP/NN extra

Or reproduce the exact pinned environment (incl. dev/test tools) with
[uv](https://docs.astral.sh/uv/):

    uv sync --extra tf --extra test

## Using the GP / neural-network models

`GPyEDS.GPAM` and `GPyEDS.nn` use the Keras-2 API. TensorFlow >= 2.16 defaults to
Keras 3, so set the following before importing them (the core modules need nothing
extra):

    export TF_USE_LEGACY_KERAS=1
