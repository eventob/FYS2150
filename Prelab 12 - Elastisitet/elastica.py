import numpy as np
import matplotlib as plt
from scipy.stats import linregress

g = 9.81


def linear_tilpasning_h(h_m):
    m = np.array([0, 0.5, 1.0, 1.5, 2, 2.5]) #, 3, 3.5])
    line = linregress(m, h_m)
    return line.slope / 1e3, line.intercept / 1e3, line.stderr / 1e3


def E_1(l, d, a, dl, dd, da):
    """
    Funksjon for å finne elastisitetsmodulen eksperimentelt ved lengde, diameter og stigningstallet A.
    Returnerer GPa
    """
    e_mod = 4 * (l ** 3) * g / (3 * np.pi * abs(a) * (d ** 4))
    d_emod = np.sqrt((da / a) ** 2 + (dl / l) ** 2 + (dd / d) ** 2) * e_mod
    return e_mod / 1e9, d_emod / 1e9


def f(l, d, m, e):
    """
    Funksjon for grunnfrekvens, bruk E funnet fra oppgave 1
    Returnerer kHz.
    """
    return np.sqrt(e * np.pi * d ** 2 / (16 * m * l)) / 1e3


def E_2(l, d, m, f):
    """
    Funksjon for elastisitetsmodulen eksperimentert ved lengde, diameter, masse og frekvens f.
    """
    return 16 * m * l * f ** 2 / (np.pi * d ** 2)


# PRELAB
# oppg 4 & 5
data = np.loadtxt("maalinger_h.dat")[:, 1]
print(linear_tilpasning_h(data))

# oppg 6
a_exp, l_exp, d_exp = -1.393e-3, 1.213, 14.91e-3
print(E_1(l_exp, d_exp, a_exp, 0.002, 0.03e-3, 0.021e-3))

# oppg 7
lengde, masse, emodul, diam = 1.530, 2.5, 107e9, 14.91e-3
print(f(lengde, diam, masse, emodul))

