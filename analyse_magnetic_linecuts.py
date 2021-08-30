import json

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import Image, perpendicular_linecuts, load_magnetic_data, normalised_gaussian, hist_and_fit_gauss

RESAMPLE_FACTOR = 20

def magnetic_edge(x, x0, Ms, theta, phi, d, t):
    u = x-x0
    u2 = u**2
    return 2 * Ms * t * ( np.sin(theta) * np.cos(phi) * d / (u**2 + d**2) - np.cos(theta) * u / (u**2 + d**2) )

def evaluate_cuts(params, lcx, lcy, t):
    x0_x, x0_y, Ms, theta, phi, rot, d, c_x, c_y = extract_params(params)
    flcx = magnetic_edge(lcx, x0_x, Ms, theta, phi, d, t) + c_x
    flcy = magnetic_edge(lcy, x0_y, Ms, theta, phi + ((-1)**int(rot))*np.pi/2, d, t) + c_y
    return flcx, flcy

def evaluate_gaussian_cuts(params, lcx, lcy, fwhm, t):
    RESAMPLE_FACTOR = 10 # to interpolate and then decimate by

    flcx, flcy = evaluate_cuts(params, lcx, lcy, t)
    
    fx = interp1d(lcx, flcx)
    fy = interp1d(lcy, flcy)

    x_smooth = np.linspace(lcx[0], lcx[-1], lcx.shape[0] * RESAMPLE_FACTOR)
    y_smooth = np.linspace(lcy[0], lcy[-1], lcy.shape[0] * RESAMPLE_FACTOR)

    dx = x_smooth[1] - x_smooth[0]
    dy = y_smooth[1] - y_smooth[0]

    kernel_x = normalised_gaussian(x_smooth-(lcx[0]+lcx[-1])/2, fwhm)
    kernel_y = normalised_gaussian(y_smooth-(lcy[0]+lcy[-1])/2, fwhm)

    cflcx_smooth = np.convolve(fx(x_smooth), kernel_x, mode='same') * dx
    cflcy_smooth = np.convolve(fy(y_smooth), kernel_y, mode='same') * dy

    # from scipy.integrate import trapezoid
    # print(trapezoid(x_smooth, kernel_x))
    # fig, axes = plt.subplots(2, 3)
    # axes[0][0].plot(fx(x_smooth), label="edge")
    # axes[0][1].plot(cflcx_smooth, label="edge*G")
    # axes[0][2].plot(x_smooth, kernel_x, label="G")
    # axes[1][0].plot(fy(y_smooth), label="edge")
    # axes[1][1].plot(cflcy_smooth, label="edge*G")
    # axes[1][2].plot(y_smooth, kernel_y, label="G")
    # [[ax.legend() for ax in row] for row in axes]
    # plt.show()
    # quit()

    # now decimate
    cflcx = cflcx_smooth[::RESAMPLE_FACTOR]
    cflcy = cflcy_smooth[::RESAMPLE_FACTOR]

    return cflcx, cflcy


def evaluate_gaussian_layer_cuts(params, lcx, lcy, fwhm, t, NVt):
    
    x0_x, x0_y, Ms, theta, phi, rot, d, c_x, c_y = extract_params(params)

    z = np.linspace(d, d+NVt, 10)
    fx = np.zeros((z.size, lcx.size))
    fy = np.zeros((z.size, lcy.size))

    for i in range(len(z)):
        _params = compact_params(x0_x, x0_y, Ms, theta, phi, rot, z[i], c_x, c_y)
        fx[i], fy[i] = evaluate_gaussian_cuts(_params, lcx, lcy, fwhm, t)

    return trapz(fx, z, axis=0) / NVt, trapz(fy, z, axis=0) / NVt


def extract_params(params):
    x0_x = params[0]
    x0_y = params[1]
    Ms = params[2]
    theta = params[3]
    phi = params[4]
    rot = params[5]
    d = params[6]
    c_x = params[7]
    c_y = params[8]
    return x0_x, x0_y, Ms, theta, phi, rot, d, c_x, c_y


def compact_params(x0_x, x0_y, Ms, theta, phi, rot, d, c_x, c_y):
    return [x0_x, x0_y, Ms, theta, phi, rot, d, c_x, c_y]


def two_cut_residual(params, lcx, lcxv, lcy, lcyv, t):
    flcx, flcy = evaluate_cuts(params, lcx, lcy, t)
    return np.concatenate([lcxv-flcx, lcyv-flcy])


def two_cut_gaussian_residual(params, lcx, lcxv, lcy, lcyv, fwhm, t):
    cflcx, cflcy = evaluate_gaussian_cuts(params, lcx, lcy, fwhm, t)
    return np.concatenate([lcxv-cflcx, lcyv-cflcy])


def two_cut_gaussian_layer_residual(params, lcx, lcxv, lcy, lcyv, fwhm, t, NVt):
    icflcx, icflcy = evaluate_gaussian_layer_cuts(params, lcx, lcy, fwhm, t, NVt)
    return np.concatenate([lcxv-icflcx, lcyv-icflcy])


def get_bounds(lcx, lcy):

    lower_bounds = compact_params(
        x0_x=lcx[0],
        x0_y=lcy[0],
        Ms=-1,
        theta=0,
        phi=-180*np.pi/180,
        rot=-np.inf,
        d=0,
        c_x=-np.inf,
        c_y=-np.inf
    )

    upper_bounds = compact_params(
        x0_x=lcx[-1],
        x0_y=lcy[-1],
        Ms=1,
        theta=90*np.pi/180,
        phi=180*np.pi/180,
        rot=np.inf,
        d=np.inf,
        c_x=np.inf,
        c_y=np.inf
    )

    bounds = np.zeros((2, 9))
    bounds[0] = lower_bounds
    bounds[1] = upper_bounds

    return bounds


def get_x_guess(lcx, lcy):
    return compact_params(
        x0_x=lcx[len(lcx)//2],
        x0_y=lcy[len(lcy)//2],
        Ms=1e-2,
        theta=30*np.pi/180,
        phi=0,
        rot=0,
        d=3e-6,
        c_x=0,
        c_y=0
    )


def fit_magnetic_edge(lcx, lcxv, lcy, lcyv, t):

    result = least_squares(two_cut_residual, args=(
        lcx, lcxv, lcy, lcyv, t), x0=get_x_guess(lcx, lcy), bounds=get_bounds(lcx, lcy))

    return result


def fit_magnetic_edge_with_gaussian(lcx, lcxv, lcy, lcyv, fwhm, t):

    result = least_squares(two_cut_gaussian_residual, args=(
        lcx, lcxv, lcy, lcyv, fwhm, t), x0=get_x_guess(lcx, lcy), bounds=get_bounds(lcx, lcy))

    return result


def fit_magnetic_edge_with_gaussian_layer(lcx, lcxv, lcy, lcyv, fwhm, t, NVt):

    result = least_squares(two_cut_gaussian_layer_residual, args=(
        lcx, lcxv, lcy, lcyv, fwhm, t, NVt), x0=get_x_guess(lcx, lcy), bounds=get_bounds(lcx, lcy))

    return result


######################################################################################

def main():

    magnetic_layer_thickness = 1e-9
    NV_layer_thickness = 1e-6

    N_linecuts = 10
    linecut_width = 20e-6

    img, optical_fwhm, p1, p2, p3, p4 = load_magnetic_data("magnetic_20x.json")

    optical_fwhm_px = optical_fwhm / img.px_to_m(1) * RESAMPLE_FACTOR
    assert optical_fwhm_px > 10

    d_vals = []
    Ms_vals = []
    theta_vals = []
    phi_vals = []

    plot_every = N_linecuts**2 // 2

    pbar = tqdm(total=N_linecuts**2)
    for lcx, lcxv in perpendicular_linecuts(img, p1, p2, linecut_width, N_linecuts):
        for lcy, lcyv in perpendicular_linecuts(img, p3, p4, linecut_width, N_linecuts):
            pbar.update()
            
            # result = fit_magnetic_edge(lcx, lcxv, lcy, lcyv, magnetic_layer_thickness)
            # flcx, flcy = evaluate_cuts(result.x, lcx, lcy)

            # result = fit_magnetic_edge_with_gaussian(lcx, lcxv, lcy, lcyv, optical_fwhm, magnetic_layer_thickness)
            result = fit_magnetic_edge_with_gaussian_layer(lcx, lcxv, lcy, lcyv, optical_fwhm, magnetic_layer_thickness, NV_layer_thickness)
            
            x0_x, x0_y, Ms, theta, phi, rot, d, c_x, c_y = extract_params(result.x)

            if d < 10e-6:
                Ms_vals.append(abs(Ms))
                d_vals.append(d)
                theta_vals.append(theta)
                phi_vals.append(phi)

            if pbar.n % plot_every == 0:
                # flcx, flcy = evaluate_gaussian_cuts(result.x, lcx, lcy, optical_fwhm, magnetic_layer_thickness)
                flcx, flcy = evaluate_gaussian_layer_cuts(result.x, lcx, lcy, optical_fwhm, magnetic_layer_thickness, NV_layer_thickness)
                fig, axes = plt.subplots(1, 2)
                axes[0].plot(lcx*1e6, lcxv, 'x')
                axes[1].plot(lcy*1e6, lcyv, 'x')
                axes[0].set_xlabel('x (um)')
                axes[1].set_xlabel('y (um)')
                axes[0].plot(lcx*1e6, flcx)
                axes[1].plot(lcy*1e6, flcy)
                # print()
                # print(d)
                # plt.show()
                # quit()

    print()

    print(f"mean d = {np.mean(d_vals)*1e6:.2f}um")
    print(f"std d = {np.std(d_vals)*1e6:.2f}um")

    print(f"mean Ms = {np.mean(Ms_vals) * 1e7 / 1e6:.2e} MA/m")
    print(f"std Ms = {np.std(Ms_vals) * 1e7 / 1e6:.2e} MA/m")

    print(f"mean theta = {np.mean(theta_vals)*180/np.pi:.2f} deg")
    print(f"std theta = {np.std(theta_vals)*180/np.pi:.2f} deg")

    print(f"mean phi = {np.mean(phi_vals)*180/np.pi:.2f} deg")
    print(f"std phi = {np.std(phi_vals)*180/np.pi:.2f} deg")

    print()

    fit_d, std_d = hist_and_fit_gauss(np.array(d_vals), plot=True, title="d")
    fit_theta, std_theta = hist_and_fit_gauss(np.array(theta_vals), plot=True, title="theta")
    fit_phi, std_phi = hist_and_fit_gauss(np.array(phi_vals), plot=True, title="phi")
    fit_Ms, std_Ms = hist_and_fit_gauss(np.array(Ms_vals), plot=True, logplot=False, title="Ms")

    print(f"fit d = {fit_d*1e6:.2f} +/- {std_d*1e6:.2f} um")
    print(f"fit theta = {fit_theta*180/np.pi:.2f} +/- {std_theta*180/np.pi:.2f} deg")
    print(f"fit phi = {fit_phi*180/np.pi:.2f} +/- {std_phi*180/np.pi:.2f} deg")
    print(f"fit Ms = {fit_Ms*1e7/1e6:.2f} +/- {std_Ms*1e7/1e6:.2f} MA/m")

    plt.figure()
    plt.imshow(img.data.T, vmin=-20e-6, vmax=20e-6)
    plt.colorbar()
    plt.arrow(100, 100, 100*np.cos(fit_phi+np.pi/2), 100*np.sin(fit_phi+np.pi/2), width=10)

    plt.show()

if __name__ == "__main__":
    main()