# GPyEDS

Autoencoder-based unsupervised feature-extraction toolbox for energy-dispersive
spectroscopy (EDS) data.

## Installation

GPyEDS requires **Python 3.10-3.12** (3.12 recommended). The Gaussian-process and
neural-network models are built on `gpflux`, which caps the stack at TensorFlow 2.16
/ Python 3.12.

### Install directly from GitHub (recommended)

No clone or build artifacts needed:

    pip install "git+https://github.com/TomWilliamsBrown/GPyEDS.git"

With the optional GP / neural-network models (TensorFlow, GPflow, GPflux):

    pip install "GPyEDS[tf] @ git+https://github.com/TomWilliamsBrown/GPyEDS.git"

Pin to a tagged release for reproducible installs (recommended — see below):

    pip install "GPyEDS[tf] @ git+https://github.com/TomWilliamsBrown/GPyEDS.git@1.0.1"

`uv` users can do the same:

    uv pip install "git+https://github.com/TomWilliamsBrown/GPyEDS.git"
    # or add it as a dependency of your own project:
    uv add "GPyEDS @ git+https://github.com/TomWilliamsBrown/GPyEDS.git"

> The bare URL installs the repository's **default branch**. Make sure your migrated
> code is the default branch (e.g. `main`), or append `@<branch-or-tag>` to pick a
> specific ref.

### From a clone — exact tested environment (recommended for development / CI)

This installs every dependency at the exact versions the test suite passed against
(via the committed `uv.lock`) and fetches a matching Python automatically. Needs
[uv](https://docs.astral.sh/uv/):

    git clone https://github.com/TomWilliamsBrown/GPyEDS.git   # get the source + the committed uv.lock
    cd GPyEDS
    uv sync --extra tf --extra test                            # build the exact environment

Then run things with `uv run` (e.g. `uv run pytest`), or activate the env with
`source .venv/bin/activate`.

Prefer plain pip? An editable install that resolves dependencies fresh from PyPI:

    pip install -e '.[tf]'

## Usage

    import GPyEDS
    from GPyEDS import mean_centre

The GP and neural-network models (`GPyEDS.GPAM`, `GPyEDS.nn`) use the Keras-2 API.
TensorFlow >= 2.16 defaults to Keras 3, so set this before importing them (the core
modules need nothing extra):

    export TF_USE_LEGACY_KERAS=1

## Acknowledgements

Based on the original GPyEDS by Norbert Toth
([DOI:10.5281/zenodo.13837097](https://doi.org/10.5281/zenodo.13837097)).
Distributed under the MIT License — see [LICENSE](LICENSE).
