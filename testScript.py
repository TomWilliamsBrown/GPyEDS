import numpy as np
import pandas as pd
from GPyEDS.EDS2CHEM import mcmc
import GPyEDS
from GPyEDS import utils
import pickle
import chardet
import matplotlib.pyplot as plt
import corner
from MCMC_emcee import align


def calc_an(CaO, Na2O):
    return ((CaO / 56.0774) / (CaO / 56.0774 + 2 * Na2O / 61.9789)) * 100


def main():
    with open("/Users/tomwilliams/Desktop/all_qemscan_dict.pkl", "rb") as f:
        r = pickle.load(f)["0501_2_extra"]["plag"]

    gauss_plag_pca = utils.gaussian_filter(utils.get_img(r[0][:, 0], r[1]), r[1], std=1)

    # --- Overview image (displayed only, not saved) ---
    fig_overview, ax_overview = plt.subplots()
    ax_overview.imshow(gauss_plag_pca, interpolation="none", vmin=0, vmax=50)
    plt.show()
    plt.close(fig_overview)

    plag_crysts = ['Sample501_2_c2r_plag1', 'Sample501_2_r2c_plag2', 'Sample501_2_plag3']
    regions = [[1500, 1800, 3000, 3400], [1700, 2000, 1400, 1800], [1500, 1900, 950, 1250]]
    params_list = [[100., 225., 120., 140., 5., 1., 1., 90.],
                   [145., 240., 150., 175., 5., 1., 1., 90.],
                   [200., 175., 250., 150., 5., 1., 1., 90.]]

    with open("/Users/tomwilliams/Desktop/Sample501_2_0_oxide.txt", 'rb') as f:
        result = chardet.detect(f.read())

    DF_NEW = pd.read_csv("/Users/tomwilliams/Desktop/Sample501_2_0_oxide.txt",
                         skiprows=[i for i in range(51)], delimiter="\t",
                         encoding=result['encoding'],
                         index_col=['Comment', 'DataSet/Point'])

    DF_NEW["Anorthite"] = calc_an(DF_NEW["CaO"].to_numpy(), DF_NEW["Na2O"].to_numpy())

    for i in range(3):
        if i == 0:
            continue
        if i == 2:
            continue
        region = regions[i]
        xl_pca1 = gauss_plag_pca[region[0]:region[1], region[2]:region[3]]

        plag1 = DF_NEW.loc[plag_crysts[i]]

        params1 = np.asarray(params_list[i])

        parameter_delta = [50., 50., 50., 50.]
        pmin1 = np.array([params1[0] - parameter_delta[0], params1[1] - parameter_delta[1],
                          params1[2] - parameter_delta[2], params1[3] - parameter_delta[3],
                          1., -100., -100., 0.0])
        pmax1 = np.array([params1[0] + parameter_delta[0], params1[1] + parameter_delta[1],
                          params1[2] + parameter_delta[2], params1[3] + parameter_delta[3],
                          30., 0., 100., 180.0])

        an_1 = plag1.Anorthite.values / 100

        mc3_output1 = mcmc.MCMC_run(
            xl_pca1, an_1, an_1 * 0.01, params=params1,
            pmin=pmin1, pmax=pmax1,
            positions=np.linspace(0, len(an_1) - 1, len(an_1)).astype("int64"),
            num_iter=1000, return_=True, name=plag_crysts[i])

        samples = mc3_output1.get_chain(discard=100, thin=18, flat=True)

        np.save("july24_emcee_flat_samples_plag_" + str(i + 1) + ".npy", samples)

        # --- Corner plot (corner.corner returns its own figure; save it, then close) ---
        fig_corner = corner.corner(
            samples, labels=["x1", "y1", "x2", "y2", "w", "m", "c", "phi"],
            show_titles=True, title_fmt=".4f")
        fig_corner.savefig("plag2_corner.png", dpi=300)
        plt.close(fig_corner)

        xdata = np.linspace(-20, 50, 250)

    # NOTE: the blocks below sit outside the loop, so they use the LAST crystal
    # processed (here i == 1). If you ever enable more than one crystal, move
    # this plotting inside the loop so each crystal gets its own figures.

    # --- Calibration figure ---
    ysamples = [xdata[i] * samples[:, 5] + samples[:, 6] for i in range(len(xdata))]
    ye1 = np.quantile(np.sort(ysamples, axis=0), 0.16, axis=1)
    ye2 = np.quantile(np.sort(ysamples, axis=0), 0.84, axis=1)
    ye3 = np.quantile(np.sort(ysamples, axis=0), 0.025, axis=1)
    ye4 = np.quantile(np.sort(ysamples, axis=0), 0.975, axis=1)

    fig_cal, ax_cal = plt.subplots()
    ax_cal.plot(xdata, np.mean(ysamples, axis=1), 'k-')
    ax_cal.fill_between(xdata[::-1], ye2, ye1, alpha=0.5, color='r', label="+/- 1 SD")
    ax_cal.fill_between(xdata[::-1], ye4, ye3, alpha=0.25, color='orange', label="+/- 2 SD")
    ax_cal.set_ylabel("Anorthite content")
    ax_cal.set_xlabel("PC1 score")
    ax_cal.legend()
    fig_cal.savefig("plag2_calibration_fig.png", dpi=300)
    plt.close(fig_cal)

    # --- Model envelope figure ---
    model_res = []
    for item in samples[::10, :]:
        model = align(xl_pca1, len(an_1), item)
        model_res.append(model)
    model_res = np.asarray(model_res)

    yes1, yes2, yes3, yes4 = [], [], [], []
    for j in range(model_res.shape[1]):
        yes1.append(np.quantile(np.sort(model_res[:, j]), 0.16))
        yes2.append(np.quantile(np.sort(model_res[:, j]), 0.84))
        yes3.append(np.quantile(np.sort(model_res[:, j]), 0.025))
        yes4.append(np.quantile(np.sort(model_res[:, j]), 0.975))

    fig_mod, ax_mod = plt.subplots()
    ax_mod.plot(range(len(an_1)), an_1, 'k', label="EPMA")
    ax_mod.plot(range(len(model)), np.mean(model_res, axis=0), 'r', label="Calibrated EDS")
    ax_mod.fill_between(range(len(model)), yes1, yes2, color='r', alpha=0.35)
    ax_mod.fill_between(range(len(model)), yes3, yes4, color='r', alpha=0.15)
    ax_mod.set_ylabel("Anorthite content")
    ax_mod.set_xlabel("Distance")
    ax_mod.legend()
    fig_mod.savefig("EDS2CHEM_0424_plag2_wPhi_emcee_example.png", dpi=300)
    plt.close(fig_mod)

    # --- Individual sample draws figure ---
    fig_smp, ax_smp = plt.subplots()
    ax_smp.plot(range(len(an_1)), an_1, 'k', label="EPMA")
    inds = np.random.randint(0, model_res.shape[0], 20)
    for _ in range(20):
        ax_smp.plot(range(len(model)), model_res[inds[_], :], 'r', alpha=0.25)
    ax_smp.set_ylabel("Anorthite content")
    ax_smp.set_xlabel("Distance")
    fig_smp.savefig("plag2_samples.png", dpi=300)
    plt.close(fig_smp)

    # --- Map with sampled profile lines ---
    fig_map, ax_map = plt.subplots()
    ax_map.imshow(xl_pca1, interpolation="None", vmin=0, vmax=50)
    inds = np.random.randint(0, samples.shape[0], 1000)
    for _ in range(1000):
        item = samples[inds[_]]
        ax_map.plot([item[1], item[3]], [item[0], item[2]], 'r-', alpha=0.05)
    fig_map.savefig("plag2_map_w_samples.png", dpi=300)
    plt.close(fig_map)


if __name__ == "__main__":
    main()
