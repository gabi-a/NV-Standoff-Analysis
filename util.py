import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

# NA_from_obj = {
#     10: 0.3,
#     20: 0.45,
#     40: 0.75
# }

diffraction_limit_fwhm = {
    10: 637e-9 / (2 * 0.3),
    20: 637e-9 / (2 * 0.45)
}

def normalised_gaussian(p, fwhm):
    x = 2*p/fwhm
    out = 2**(-x**2) / (fwhm * np.sqrt(np.pi / np.log(16)))
    return out

def gaussian(p, fwhm, p0, a, c):
    return normalised_gaussian(p - p0, fwhm) * a + c

def gaussian_edge(x, fwhm, x0, a, c):
    return a * 0.5 * (1 + erf(2 * (x-x0) * np.sqrt(np.log(2)) / fwhm)) + c

def r_squared(f, xdata, ydata, popt):
    residuals = ydata - f(xdata, *popt)
    ss_res = np.sum(residuals**2)                  
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def hist_and_fit_gauss(data, plot=False, title=None, fit=True, logplot=False, save=None, histrange=None):

    u = np.nan
    fwhm = np.nan

    if fit:
        n, bins = np.histogram(data, bins='auto', range=histrange)
        u = bins[np.argmax(n)]
        fwhm = np.std(data) * 2.35
        a = np.max(n)
        try:
            popt, pcov = curve_fit(gaussian, bins[:-1], n, p0=(fwhm, u, a, 0))
            u = popt[1]
            fwhm = abs(popt[0]) / 2.355
        except RuntimeError:
            fit = False

    if plot:
        plt.figure()
        if logplot:
            plt.hist(np.log10(data), bins='auto')
        else:
            plt.hist(data, bins='auto')
            if fit:
                xsmooth = np.linspace(bins[0], bins[-2], 1000)
                plt.plot(xsmooth, gaussian(xsmooth, *popt), 'r')

        plt.title(title)

    if save is not None:
        plt.savefig(os.path.join(save, f"{title}.svg"))

    return u, fwhm

def perpendicular_linecuts(img, p1, p2, linecut_width, N_linecuts, return_center=False):

    # print(f"perpendicular_linecuts: linecut length = {2*linecut_width:.2e}")

    t = np.arctan2(-p2[0]+p1[0], p2[1]-p1[1])
    dx = linecut_width*np.cos(t)
    dy = linecut_width*np.sin(t)

    x_along_edge = np.linspace(p1[0], p2[0], N_linecuts)
    y_along_edge = np.linspace(p1[1], p2[1], N_linecuts)

    if return_center:
        for xv, yv in zip(x_along_edge, y_along_edge):
            yield img.extract_linecut([xv-dx, yv-dy], [xv+dx, yv+dy]), (xv, yv)
    else:
        for xv, yv in zip(x_along_edge, y_along_edge):
            yield img.extract_linecut([xv-dx, yv-dy], [xv+dx, yv+dy])

def load_magnetic_data(fname):
    
    data = json.load(open(fname))

    img = Image()
    img.load(data['fname'], data['objective'])

    line1 = data['line1']
    line2 = data['line2']
    p1 = (img.px_to_m(line1['p0']['x']), img.px_to_m(line1['p0']['y']))
    p2 = (img.px_to_m(line1['p1']['x']), img.px_to_m(line1['p1']['y']))
    p3 = (img.px_to_m(line2['p0']['x']), img.px_to_m(line2['p0']['y']))
    p4 = (img.px_to_m(line2['p1']['x']), img.px_to_m(line2['p1']['y']))
    
    optical_fwhm = data['optical_fwhm']

    return img, optical_fwhm, p1, p2, p3, p4

class Image:

    def __init__(self):
        self.objective = None
        self.data = None

    def px_to_m(self, val):
        return (val / self.data.shape[0]) * (self.data.shape[0] / 2048) * (11.3e-3 / self.objective)

    def get_size_m(self):
        return self.px_to_m(np.array(self.data.shape))

    def load(self, fname, objective):
        self.data = np.loadtxt(fname)
        self.objective = objective

        _x = self.px_to_m(np.arange(self.data.shape[1]))
        _y = self.px_to_m(np.arange(self.data.shape[0]))
        self.interp = interp2d(_x, _y, self.data)

    def extract_linecut(self, p1, p2, N=50):
        # extract linecut
        # p1 = (x, y) start point (m)
        # p2 = (x, y) end point (m)
        # returns
        # x - x coordinates along the linecut
        # y - y values along the linecut

        xc = np.linspace(p1[0], p2[0], N)
        yc = np.linspace(p1[1], p2[1], N)

        length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        x = np.linspace(0, length, N)

        # print(f"extract_linecut: linecut length = {length:.2e}")
    
        return x, np.diag(self.interp(xc, yc))

def peaks_widths(y_smooth, scale, height):
    peaks,_ = find_peaks(y_smooth, height=height) #, threshold=1e-6)
    results = peak_widths(y_smooth, peaks, rel_height=0.5)
    widths = results[0] * scale
    return peaks,results,widths