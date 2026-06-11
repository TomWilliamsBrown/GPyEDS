#!/usr/bin/env python3
"""
Build a QEMSCAN hyperspectral cube from an exported QEMSCAN CSV file.

This is a terminal-runnable conversion of the `QEMSCAN_build_data.ipynb` notebook.
It loads the desired element columns from a QEMSCAN CSV, builds a 3D concentration
map (hypercube) using GPyEDS.utils.build_conc_map, optionally saves preview images
of individual element layers, and writes the cube to disk as .npz and/or .pkl.

Example
-------
    python qemscan_build_data.py --elements Ca Si Fe Mg O \
        --outdir ./output --save-format both --preview

The input CSV path is hardcoded near the top of this file (see CSV_PATH).

Requirements: pandas, numpy, matplotlib, and GPyEDS (with its `utils` module).
"""

import argparse
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Hardcoded path to the exported QEMSCAN CSV file. Edit this to point at your
# own file.
# ---------------------------------------------------------------------------
CSV_PATH = "/Users/tomwilliams/Desktop/KRA13B_10mu_allphases.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a QEMSCAN hyperspectral cube from an exported CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=["Ca", "Si", "Fe", "Mg", "O"],
        help="Element columns to load and turn into the hypercube. "
        "X and Y are always added automatically as coordinate columns.",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=1,
        help="Row index (0-based) of the header line in the CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to write the output cube (and preview images) to.",
    )
    parser.add_argument(
        "--prefix",
        default="conc_map",
        help="Base name for the saved output files.",
    )
    parser.add_argument(
        "--save-format",
        choices=["npz", "pkl", "both"],
        default="npz",
        help="Format(s) to save the concentration map in.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load only the first few rows first and print the columns "
        "for inspection before doing the full load.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save PNG preview images of each element layer.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display preview images interactively instead of just saving them "
        "(requires a display / GUI backend).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Import heavy / optional dependencies inside main so that --help works
    # even if they are not installed.
    import pandas as pd
    import numpy as np

    # Use a non-interactive backend unless the user explicitly wants to view plots.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from GPyEDS import utils
    except ImportError as exc:
        sys.exit(
            f"Error importing GPyEDS: {exc}\n"
            "Make sure the GPyEDS package (with its `utils` module) is installed."
        )

    if not os.path.isfile(CSV_PATH):
        sys.exit(f"CSV file not found: {CSV_PATH}")

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional: load a small portion of the data for inspection.
    # ------------------------------------------------------------------
    if args.inspect:
        print("Loading first 10 rows for inspection...")
        data = pd.read_csv(CSV_PATH, nrows=10, header=args.header_row)
        print("\nHead of data:")
        print(data.head())
        print("\nAvailable columns:")
        print(data.columns.to_numpy())
        print()

    # ------------------------------------------------------------------
    # Load only the wanted columns.
    # Only columns containing numbers should be loaded; everything apart
    # from X and Y will be turned into the hypercube.
    # ------------------------------------------------------------------
    props = ["X", "Y"] + list(args.elements)
    print(f"Loading columns {props} from {CSV_PATH} ...")
    data_wanted = pd.read_csv(CSV_PATH, header=args.header_row, usecols=props)
    print("\nHead of loaded data:")
    print(data_wanted.head())

    # ------------------------------------------------------------------
    # Build the concentration map (hypercube).
    # ------------------------------------------------------------------
    print("\nBuilding concentration map...")
    conc_map, data_mask = utils.build_conc_map(data_wanted)
    print(f"Concentration map shape: {conc_map.shape}")
    print(f"Data mask shape: {data_mask.shape}")

    # Elements correspond to the columns after X and Y.
    elements = props[2:]

    # ------------------------------------------------------------------
    # Optional: preview each element layer.
    # ------------------------------------------------------------------
    if args.preview or args.show:
        cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        for i, element in enumerate(elements):
            cmap = cmaps[i % len(cmaps)]
            plt.figure()
            plt.imshow(conc_map[:, :, i], interpolation="none", cmap=cmap)
            plt.title(element)
            plt.colorbar()
            if args.preview:
                img_path = os.path.join(args.outdir, f"{args.prefix}_{element}.png")
                plt.savefig(img_path, bbox_inches="tight", dpi=150)
                print(f"Saved preview: {img_path}")
        if args.show:
            plt.show()
        plt.close("all")

    # ------------------------------------------------------------------
    # Save the cube.
    # ------------------------------------------------------------------
    if args.save_format in ("npz", "both"):
        npz_path = os.path.join(args.outdir, f"{args.prefix}.npz")
        np.savez(
            npz_path,
            conc_map=conc_map,
            data_mask=data_mask,
            elements=elements,
        )
        print(f"Saved: {npz_path}")

    if args.save_format in ("pkl", "both"):
        pkl_path = os.path.join(args.outdir, f"{args.prefix}.pkl")
        to_save = {
            "conc_map": conc_map,
            "data_mask": data_mask,
            "elements": elements,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(to_save, f)
        print(f"Saved: {pkl_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
