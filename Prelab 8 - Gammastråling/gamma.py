import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

poisson = np.loadtxt("poisson.csv")
spektrum = np.loadtxt("spektrum.csv")
# spm 2
print(np.mean(poisson))     # korrekt


def std_dev(arr):
    # spm 3
    N = len(arr)
    avg_k = (1/N) * np.sum(arr)
    return np.sqrt((1 / (N - 1)) * np.sum((arr - avg_k) ** 2))


print(std_dev(poisson))     # korrekt


def gm_eff(n_r, n_b, A, r, d):
    # spm 4
    omega = np.pi * r ** 2 / d ** 2
    return ((n_r - n_b) / (A * (omega / (4 * np.pi)))) * 1e2


print('GM_eff: %g' % gm_eff(23, 2, 1e6, 0.02, 0.2))     # korrekt


def lambert_lov(skj_arr, n_arr):
    # spm 5, 6
    plt.plot(skj_arr, n_arr)
    plt.grid()
    line = linregress(skj_arr, n_arr)
    y_line = (line.slope * skj_arr) + line.intercept
    plt.plot(skj_arr, y_line)
    plt.show()
    return abs(line.slope) / 1e3, line.stderr / 1e3


skj_x = np.array([0, 4, 8, 12, 16, 20, 24]) / 1e3           # [m]
tell_n = np.array([13.7, 12.4, 11.0, 9.7, 8.9, 7.9, 7.1])   # [s^-1]
mu, delta_mu = lambert_lov(skj_x, tell_n)
print("mu: %g, delta_mu: %g" % (mu, delta_mu))  # feil (-275.9), feil (275.9), feil (0.276), feil (13.36), feil (13.35), feil (0.013)

# spm 7
z = abs(np.log(0.05) / 20) * 1e3
print("Tykkelse z: %g" % z)     # korrekt

# spm 8
delta_z = (0.04 * z)
print("delta_z: %g" % delta_z)      # feil (1 / 0.08), korrekt (0.04 * z)


def gamma_energi(i_arr, e_arr):
    # spm 9, 10
    line = linregress(i_arr, e_arr)
    return line.slope, line.intercept


i = np.array([410, 773])
e = np.array([662, 1275])
print("stig: %g, skjer: %g" % gamma_energi(i, e))   # feil (1.7, 2), korrekt (1.69), korrekt (-30)

# spm 11
kanal = np.linspace(0, len(spektrum), len(spektrum))
fwhm_y = np.max(spektrum) / 2 + (35 / 2)    # legger til lineær nullpunktsenergi
fwhm_x = kanal[np.logical_and(spektrum < fwhm_y + 2, spektrum > fwhm_y - 2)]
fwhm = (fwhm_x[1] - fwhm_x[0]) * 2
print("FWHM keV: %g" % fwhm)        # feil (243.3), feil (243), korrekt (264)

plt.plot(kanal, spektrum)
plt.axhline(fwhm_y, color='red', linestyle='--')
plt.axhline(np.max(spektrum), color='orange', linestyle='dotted')
plt.axvline(fwhm_x[1], 0, fwhm_y / np.max(spektrum), color='red', linestyle='--')
plt.axvline(fwhm_x[0], 0, fwhm_y / np.max(spektrum), color='red', linestyle='--')
plt.xlabel("Kanal"), plt.ylabel("Tellinger/kanal [n / kanal]")
plt.grid()
plt.show()
