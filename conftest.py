import os

# Use a non-interactive matplotlib backend so the plotting tests run headless.
os.environ.setdefault("MPLBACKEND", "Agg")
