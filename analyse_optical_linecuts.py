import os

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from util import Image, diffraction_limit_fwhm, gaussian_edge, r_squared, hist_and_fit_gauss, perpendicular_linecuts

root = "F:/ExperimentalData/2021/2021-06/Optical_Resolution"
objectives = [10, 20]
configs = {
    "NV_under_ant": "direct",
    "NV_under_ant_thruglass": "through glass",
    "NV_up_on_ant": "through diamond"
}

linecut_width = 20e-6  # 10um
N_linecuts = 300

results = pd.DataFrame(columns=["objective", "configuration", "diffraction limit (FWHM)", "measured (FWHM)"])

for objective in objectives:

    diff_lim_fwhm = diffraction_limit_fwhm[objective]

    print(f"diffraction limit (FWHM) = {diff_lim_fwhm:.2e} m")

    for config in configs:

        path = os.path.join(root, f"{objective}x", "BrokenAntenna_179", config)
        files = []

        for (_, _, fnames) in os.walk(path):
            for fname in fnames:
                if fname[-3:] != "npy":
                    continue
                files.append(os.path.join(path, fname))

        measured_fwhm = []

        for fpath in tqdm(files):

            info = np.load(fpath, allow_pickle=True).item()

            img = Image()
            img.load(info['fname'], objective)

            p1, p2 = info['points']
            p1 = [img.px_to_m(a) for a in p1]  # convert to m
            p2 = [img.px_to_m(a) for a in p2]  # convert to m

            for lcx, lcv in perpendicular_linecuts(img, p1, p2, linecut_width, N_linecuts):
                try:
                    popt, pcov = curve_fit(gaussian_edge, lcx, lcv,
                                        (diff_lim_fwhm,
                                            lcx[len(lcx)//2],
                                            max(lcv)-min(lcv),
                                            np.mean(lcx)),
                                        maxfev=2000)
                except RuntimeError:
                    continue
                    
                # plt.figure()
                # plt.plot(lcx, lcv, 'x')
                # plt.plot(lcx, gaussian_edge(lcx, *popt))
                # plt.show()
                # print(f"FWHM = {popt[0]:0.2e}")

                if r_squared(gaussian_edge, lcx, lcv, popt) > 0.9:
                    measured_fwhm.append(abs(popt[0]))
            
        # plt.figure()
        # plt.hist(measured_fwhm, bins=20)
        # plt.show()

        est_fwhm, err_fwhm = hist_and_fit_gauss(measured_fwhm, plot=True, histrange=(0, 5*diff_lim_fwhm))
        # plt.show()
        print(f"Estimated FWHM = {est_fwhm:.2e} +/- {err_fwhm:.2e}")

        results = results.append({
            "objective":objective, 
            "configuration":configs[config], 
            "diffraction limit (FWHM)":diff_lim_fwhm, 
            "measured (FWHM)":est_fwhm,
            "error (1 sigma)":err_fwhm
        }, ignore_index=True)

results.to_pickle("optical_results.df")
print(results)