import numpy as np
import matplotlib.pyplot as plt

def lorentzian(p, w):
    x = 2*p/w
    return 1/(1+x**2)

def Bz(p, w):
    x = 2*p/w
    return x/(1+x**2)

x = np.linspace(-10, 10)

def test_convolve_lorentzians():
    y1 = lorentzian(x, 2)
    y2 = lorentzian(x, 3)
    y3 = lorentzian(x, 5)
    y4 = np.convolve(y1, y2, mode='same')
    y4 /= np.max(y4)
    y5 = np.convolve(y2, y1, mode='same')
    y5 /= np.max(y5)

    plt.plot(x, y1, label="L(2)")
    plt.plot(x, y2, label="L(3)")
    plt.plot(x, y3, label="L(5)")
    plt.plot(x, y4, '-o', label="L(2)*L(3)")
    plt.plot(x, y5, '-x', label="L(3)*L(2)")
    plt.legend()
    plt.show()

def test_convolve_lorentzian_with_Bz():
    y1 = Bz(x, 2)
    y2 = lorentzian(x, 3)
    y3 = Bz(x, 5); y3 /= np.max(y3)
    y4 = np.convolve(y1, y2, mode='same'); y4 /= np.max(y4)
    plt.plot(x, y3, label='Bz(5)')
    plt.plot(x, y4, label='Bz(2)*L(3)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # test_convolve_lorentzians()
    test_convolve_lorentzian_with_Bz()