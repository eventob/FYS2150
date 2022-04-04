import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

poisson = np.loadtxt("poisson.csv")
spektrum = np.loadtxt("spektrum.csv")
print(np.mean(poisson))


def std_dev(arr):
    N = len(arr)
    avg_k = (1/N) * np.sum(arr)
    return np.sqrt((1 / (N - 1)) * np.sum((arr - avg_k) ** 2))


print(std_dev(poisson))


def gm_eff(n_r, n_b, A, r, d):
    omega = 4 * np.pi * (r * d)
    print(omega)
    return ((n_r - n_b) / (A * (omega / (4 * np.pi)))) * 1e2


print(gm_eff(23, 2, 1e6, 0.002, 0.2))


def lambert_lov(I_I0, arr_x):
    skj_sum = np.sum(I_I0)
    x_sum = np.sum(arr_x)
    return np.log(skj_sum) / x_sum


skj_x = np.array([0, 4, 8, 12, 16, 20, 24]) / 1e3     # [m]
tell_n = np.array([13.7, 12.4, 11.0, 9.7, 8.9, 7.9, 7.1])
mu = lambert_lov(skj_x, tell_n)
delta_mu = np.sqrt((std_dev(skj_x) / np.sum(skj_x)) ** 2 + (std_dev(tell_n) / np.sum(tell_n)) ** 2) * mu
print(mu, delta_mu)


def gamma_energi(I, E):
    delta_e = E / I
    e_0 = E - (delta_e * I)
    return delta_e, e_0


print(gamma_energi(410, 622), gamma_energi(773, 1275))

kanal = np.linspace(0, len(spektrum), len(spektrum))
y_fwhm = np.max(spektrum) / 2
fwhm_points = kanal[np.logical_and(spektrum < (y_fwhm + 2), spektrum > (y_fwhm - 2))]
fwhm = fwhm_points[1] - fwhm_points[0]
print(fwhm)

plt.plot(kanal, spektrum)
plt.axhline(y_fwhm, color='red', linestyle='--')
plt.show()
