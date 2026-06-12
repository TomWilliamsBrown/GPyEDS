#!/usr/bin/env python3
"""
Segmentation + post-processing example for GPyEDS hyperspectral data.

This is a terminal-runnable conversion of the `Segmentation.ipynb` notebook.
It loads a set of per-element CSV maps, stacks them into a concentration cube,
masks out empty regions, removes a principal component, trains a two-layer GPAM
model, clusters the latent space with K-means, and runs PCA on a single phase.

All inline notebook displays are turned into printed output, and every plot is
saved as a PNG into the output directory (use --show to display interactively).

Example
-------
    python segmentation.py --outdir ./output --epochs 10 --clusters 3

The input data directory is hardcoded near the top of this file (see DATA_DIR).
Requirements: GPyEDS, numpy, pandas, matplotlib, scikit-learn.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Hardcoded location of the per-element CSV maps. Each element is expected at
#   {DATA_DIR}/Montaged Map Data-{element} K series.csv
# Edit DATA_DIR (and FILENAME_TEMPLATE if your naming differs) to suit.
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/tomwilliams/Desktop/MontageEDS"
FILENAME_TEMPLATE = "Montaged Map Data-{element} K series.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Segmentation + post-processing example for GPyEDS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=["Al", "Mg", "Ca", "Si", "O", "Fe", "Na", "Cr"],
        help="Elements to load and stack into the concentration cube.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to write output figures to.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=400,
        help="Minimum summed counts for a pixel to be kept in the mask.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=6,
        help="Number of components for the initial decompose inspection.",
    )
    parser.add_argument(
        "--comp-to-remove",
        type=int,
        default=0,
        help="Principal component index to remove from the cube.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for the GPAM model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Training batch size for the GPAM model.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of K-means clusters for segmentation.",
    )
    parser.add_argument(
        "--phase-label",
        type=int,
        default=1,
        help="Cluster label to isolate for the PCA step.",
    )
    parser.add_argument(
        "--filter-range",
        type=int,
        default=2,
        help="Range for the spatial smoothing filter on the PCA scores.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively instead of saving them to PNG.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import numpy as np
    import pandas as pd

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from GPyEDS import utils, GPAM, spatial_filters
    except ImportError as exc:
        sys.exit(
            f"Error importing GPyEDS: {exc}\n"
            "Make sure the GPyEDS package is installed."
        )
    from sklearn.cluster import KMeans

    os.makedirs(args.outdir, exist_ok=True)

    def output_figs(name):
        """Save (or show) every currently-open matplotlib figure, then close.

        Handles both figures we create here and figures created internally by
        functions like utils.decompose(..., plot=True).
        """
        fignums = plt.get_fignums()
        if args.show:
            plt.show()
        else:
            for n in fignums:
                fig = plt.figure(n)
                suffix = "" if len(fignums) == 1 else f"_{n}"
                path = os.path.join(args.outdir, f"{name}{suffix}.png")
                fig.savefig(path, bbox_inches="tight", dpi=150)
                print(f"Saved figure: {path}")
        plt.close("all")

    elements = args.elements

    # ------------------------------------------------------------------
    # Load data: one CSV per element, stacked into a concentration cube.
    #
    # Most data should be straightforward to load with Hyperspy instead;
    # extract the underlying NumPy array from the Hyperspy object and use
    # it in place of conc_map here.
    # ------------------------------------------------------------------
    print(f"Loading {len(elements)} element maps from '{DATA_DIR}' ...")
    maps = []
    for item in elements:
        path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(element=item))
        if not os.path.isfile(path):
            sys.exit(f"CSV file not found: {path}")
        maps.append(pd.read_csv(path).to_numpy())

    # Stack the individual element maps along a new spectral axis.
    conc_map = np.zeros((maps[0].shape[0], maps[0].shape[1], len(maps)))
    for i in range(len(maps)):
        conc_map[:, :, i] += maps[i]
    print(f"Concentration cube shape: {conc_map.shape}")

    # ------------------------------------------------------------------
    # Inspect counts summed across the spectral dimension.
    # ------------------------------------------------------------------
    m = np.sum(conc_map, axis=-1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(m)
    ax[1].hist(m.ravel(), 100)
    output_figs("01_summed_counts")

    # ------------------------------------------------------------------
    # Mask: keep any pixel above the threshold. This eliminates the large
    # regions where no sample is present.
    # ------------------------------------------------------------------
    mask = m > args.mask_threshold
    plt.imshow(mask)
    output_figs("02_mask")
    print(f"Pixels kept by mask: {int(mask.sum())} / {mask.size}")

    # ------------------------------------------------------------------
    # Initial decomposition for inspection (uses a dummy all-true mask).
    # ------------------------------------------------------------------
    dummy_mask = np.ones((conc_map.shape[0], conc_map.shape[1]), dtype="bool")
    utils.decompose(
        np.nan_to_num(conc_map),
        n_components=args.n_components,
        data_mask=dummy_mask,
        plot=True,
        elements=elements,
    )
    output_figs("03_decompose_inspection")

    # Remove the chosen principal component from the cube.
    conc_map = utils.remove_pc_comp(conc_map, comp2remove=args.comp_to_remove)

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(conc_map[:, :, 5], interpolation="none")
    output_figs("04_component_removed_layer")

    # ------------------------------------------------------------------
    # Segmentation: normalise the masked data and build a GPAM model.
    # ------------------------------------------------------------------
    array = np.nan_to_num(conc_map[mask.astype("bool")])
    array_norm, params = utils.feature_normalisation(array[::1, :], return_params=True)

    model, encoder, decoder = GPAM.create_two_layer_GPAM_from_data(
        array_norm, return_layers=True
    )

    # ------------------------------------------------------------------
    # Model training.
    # ------------------------------------------------------------------
    print(f"Training GPAM model for {args.epochs} epochs ...")
    history = model.fit(
        {"inputs": array_norm[::10, :], "targets": array_norm[::10, :]},
        epochs=int(args.epochs),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Inference: project all pixels into the latent space.
    z, v = GPAM.model_inference(array_norm, encoder)

    plt.scatter(z[::10, 0], z[::10, 1], s=0.0001)
    output_figs("05_latent_scatter")

    intensity = np.histogram2d(z[:, 1], z[:, 0], bins=1000, density=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(-intensity[0] / np.sum(intensity[0]), vmin=-0.0000195, cmap="magma")
    plt.gca().invert_yaxis()
    fig.tight_layout()
    output_figs("06_latent_density")

    # ------------------------------------------------------------------
    # Clustering with K-means on the latent representation.
    # ------------------------------------------------------------------
    print(f"Clustering latent space into {args.clusters} clusters ...")
    kmm = KMeans(n_clusters=args.clusters).fit(z[::10])
    l = kmm.predict(z)

    plt.scatter(
        z[::10, 0], z[::10, 1], c=l[::10], s=0.001, cmap="tab10", vmin=0, vmax=9
    )
    output_figs("07_cluster_scatter")

    plt.figure(figsize=(12, 12))
    plt.imshow(
        utils.get_img(l.astype("float32"), mask.astype("float32")),
        interpolation="none",
        cmap="tab10",
        vmin=0,
        vmax=9,
    )
    output_figs("08_cluster_map")

    # ------------------------------------------------------------------
    # PCA on a single phase to showcase compositional variability.
    # ------------------------------------------------------------------
    phase = utils.get_img(l == args.phase_label, mask)
    phase[~mask] = 0
    plt.imshow(phase)
    output_figs("09_phase_mask")

    scores, comps = utils.decompose(
        conc_map[phase.astype("bool")],
        n_components=3,
        data_mask=phase,
        plot=True,
        elements=elements,
    )
    output_figs("10_phase_decompose")
    print(f"Scores shape: {scores.shape}")

    # Take the first component (An-Ab zoning in plagioclase in the notebook).
    plt.hist(scores[:, :, 0].ravel(), 100)
    output_figs("11_score_histogram")

    plt.imshow(scores[:, :, 0], interpolation="none", vmin=-50, vmax=50)
    output_figs("12_score_component0")

    # The raw component is noisy; apply a spatial smoothing filter.
    fig = plt.figure(figsize=(12, 12))
    filt = spatial_filters.linear_filter(
        scores[:, :, 0], mask=phase, range_=args.filter_range, type_="gaussian"
    )
    plt.imshow(filt, interpolation="none", vmin=-50, vmax=50)
    output_figs("13_score_component0_filtered")

    print("\nDone.")


if __name__ == "__main__":
    main()
