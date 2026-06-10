import os

# gpflow / gpflux and GPyEDS.nn are written against Keras 2. TensorFlow >= 2.16
# ships Keras 3 by default, so force the legacy tf-keras backend before any test
# module (and therefore TensorFlow) is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Use a non-interactive matplotlib backend so the plotting tests run headless.
os.environ.setdefault("MPLBACKEND", "Agg")
