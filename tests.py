import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from analyse_magnetic_linecuts import magnetic_edge
from util import Image, normalised_gaussian, peaks_widths

class TestGaussianFunctions(unittest.TestCase):

    def test_gaussian_fwhm(self):
        fwhm = 3
        y0 = normalised_gaussian(0, fwhm)
        y1 = normalised_gaussian(-fwhm/2, fwhm)
        y2 = normalised_gaussian(fwhm/2, fwhm)

        self.assertTrue(np.allclose([y0/2, y0/2], [y1, y2]))

    def test_gaussian_integral(self):
        fwhm = 3
        out = quad(normalised_gaussian, -np.inf, np.inf, args=(fwhm, ))[0]
        self.assertTrue(np.allclose(out, 1))

class TestMagneticFunctions(unittest.TestCase):

    def test_lorentzian_fwhm(self):
        x0 = 0
        Ms = 1
        theta = np.pi/2
        phi = 0
        d = 2
        t = 1e-3
        y0 = magnetic_edge(0, x0, Ms, theta, phi, d, t)
        y1 = magnetic_edge(-d, x0, Ms, theta, phi, d, t)
        y2 = magnetic_edge(d, x0, Ms, theta, phi, d, t)

        self.assertTrue(np.allclose([y0/2, y0/2], [y1, y2]), f"Failed Bx")

        theta = 0
        y0 = magnetic_edge(0, x0, Ms, theta, phi, d, t)
        y1 = magnetic_edge(-d, x0, Ms, theta, phi, d, t)
        y2 = magnetic_edge(d, x0, Ms, theta, phi, d, t)

        self.assertTrue(np.allclose([y0, y1, y2], [0, Ms*t/d, -Ms*t/d]), f"Failed Bz")

class TestMagnetic10xImage: # (unittest.TestCase):

    def setUp(self):
        fname = r"F:\ExperimentalData\2021\2021-05\W-CoFeB-MgO\#177k_on_arm\ODMR - CW_90_Full_bin_0\data\diffSubPlane.txt"
        self.img = Image()
        self.img.load(fname, 10)

    def test_px_to_m(self):
        self.assertTrue(np.allclose(self.img.px_to_m(1024), 565e-6), f"{self.img.px_to_m(1024):.2e} != {565e-6:.2e}")
        self.assertTrue(np.allclose(1024/565e-6, 1/self.img.px_to_m(1)), f"{1024/565e-6:.2e} != {1/self.img.px_to_m(1):.2e}")
        # fname = r"F:\ExperimentalData\2021\2021-06\Optical_Resolution\20x\BrokenAntenna_179\NV_up_on_ant\PL_image_1.txt"
        # img = Image()
        # img.load(fname, 20)
        # assert np.allclose(img.px_to_m(1024), 565e-6/2)

    def test_linecut(self):
        x, lc = self.img.extract_linecut((0, 0), (565e-6, 0))
        self.assertTrue(np.allclose(x[0], 0))
        self.assertTrue(np.allclose(x[-1], 565e-6))

    def test_plot_linecut(self):

        p1 = (220e-6, 27e-6)
        p2 = (220e-6, 84e-6)

        x, lc = self.img.extract_linecut(p1, p2)
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(self.img.data, vmin=-20e-6, vmax=20e-6)
        px_per_m = 1 / self.img.px_to_m(1)
        axes[0].plot([p1[0]*px_per_m, p2[0]*px_per_m], [p1[1]*px_per_m, p2[1]*px_per_m], 'r')
        
        axes[1].plot(x*1e6, lc)
        axes[1].set_xlabel("x (um)")

        scale = (x[-1]-x[0]) / len(x)
        peaks, results, widths = peaks_widths(-lc, scale, 3e-5)
        axes[1].plot(x[peaks]*1e6, lc[peaks], 'rx')
        axes[1].hlines(-results[1], results[2]*scale*1e6, results[3]*scale*1e6, 'C3')
        axes[1].text(results[3][0]*scale*1e6, -results[1][0], f"fwhm={widths[0]*1e6:.2f}um")

        plt.show()

class TestMagnetic20xImage(unittest.TestCase):

    def setUp(self):
        fname = r"C:\Users\Gaming\Desktop\ODMR - CW_694_Full_bin_1\data\diffSubPlane.txt"
        self.img = Image()
        self.img.load(fname, 20)

    def test_px_to_m(self):
        self.assertTrue(np.allclose(self.img.px_to_m(1024), 565e-6/2), f"{self.img.px_to_m(1024):.2e} != {565e-6/2:.2e}")
        self.assertTrue(np.allclose(1024/(565e-6/2), 1/self.img.px_to_m(1)), f"{1024/(565e-6/2):.2e} != {1/self.img.px_to_m(1):.2e}")
        # fname = r"F:\ExperimentalData\2021\2021-06\Optical_Resolution\20x\BrokenAntenna_179\NV_up_on_ant\PL_image_1.txt"
        # img = Image()
        # img.load(fname, 20)
        # assert np.allclose(img.px_to_m(1024), 565e-6/2)

    def test_linecut(self):
        x, lc = self.img.extract_linecut((0, 0), (565e-6/2, 0))
        self.assertTrue(np.allclose(x[0], 0))
        self.assertTrue(np.allclose(x[-1], 565e-6/2))

    def test_plot_linecut(self):
        
        p1 = (163e-6, 142e-6)
        p2 = (160e-6, 205e-6)

        x, lc = self.img.extract_linecut(p1, p2)
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(self.img.data, vmin=-20e-6, vmax=20e-6)
        px_per_m = 1 / self.img.px_to_m(1)
        axes[0].plot([p1[0]*px_per_m, p2[0]*px_per_m], [p1[1]*px_per_m, p2[1]*px_per_m], 'r')
        
        axes[1].plot(x*1e6, lc)
        axes[1].set_xlabel("x (um)")

        scale = (x[-1]-x[0]) / len(x)
        peaks, results, widths = peaks_widths(lc, scale, 3e-5)
        axes[1].plot(x[peaks]*1e6, lc[peaks], 'rx')
        axes[1].hlines(results[1], results[2]*scale*1e6, results[3]*scale*1e6, 'C3')
        axes[1].text(results[3][0]*scale*1e6, results[1][0], f"fwhm={widths[0]*1e6:.2f}um")

        plt.show()

class TestPL10xImage: # (unittest.TestCase):

    def setUp(self):
        fname = r"F:\ExperimentalData\2021\2021-06\Optical_Resolution\10x\BrokenAntenna_179\NV_up_on_ant\PL_image_38.txt"
        self.img = Image()
        self.img.load(fname, 10)
    
    def test_plot_linecut(self):

        p1 = (550e-6, 689e-6)
        p2 = (590e-6, 663e-6)

        x, lc = self.img.extract_linecut(p1, p2)
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(self.img.data)
        px_per_m = 1 / self.img.px_to_m(1)
        axes[0].plot([p1[0]*px_per_m, p2[0]*px_per_m], [p1[1]*px_per_m, p2[1]*px_per_m], 'r')
        
        axes[1].plot(x*1e6, lc)
        axes[1].set_xlabel("x (um)")
        axes[1].hlines([700], 22-4.4/2, 22+4.4/2, 'r')
        axes[1].text(24, 705, "width = 4.4um")

        # scale = (x[-1]-x[0]) / len(x)
        # peaks, results, widths = peaks_widths(-lc, scale, 3e-5)
        # axes[1].plot(x[peaks]*1e6, lc[peaks], 'rx')
        # axes[1].hlines(-results[1], results[2]*scale*1e6, results[3]*scale*1e6, 'C3')
        # axes[1].text(results[3][0]*scale*1e6, -results[1][0], f"fwhm={widths[0]*1e6:.2f}um")

        plt.show()


if __name__ == "__main__":

    unittest.main()
