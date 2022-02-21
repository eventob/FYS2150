import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import probplot
from scipy.stats import linregress

data = np.loadtxt("maalinger_deformasjon.dat")

m = data[:, 0]      # massen til loddet [kg]
h = data[:, 1]      # utslaget til måleuret [mm]


# oppgave 1
def best_straight_line(x, y):
    """
    Funksjon som tilpasser den beste rette linjen til dataene (y(x) = alpha * x + beta)
    """
    N = len(x)

    D = np.sum(x ** 2) - ((1 / N) * (np.sum(x) ** 2))
    E = np.sum(x * y) - ((1 / N) * np.sum(x) * np.sum(y))
    F = np.sum(y ** 2) - ((1 / N) * (np.sum(y) ** 2))

    alpha = E / D
    beta = np.mean(y) - (alpha * np.mean(x))

    d_alpha = np.sqrt(((1 / (N - 2)) * ((D * F - E ** 2) / D ** 2)))

    return alpha, beta, d_alpha


print(best_straight_line(m, h))


# oppgave 4
def fjaer_konst(arr, mass):
    """
    Funksjon som finner fjærkonstanten til snoren loddet henger i ved hjelp av masse (kg/mass) og periodetid (tau/arr)
    """
    return (2 * np.pi * np.sqrt(mass) / np.mean(arr)) ** 2


tau = [4.12, 4.04, 4.16, 4.02, 4.03, 4.04, 3.89, 4.2, 4.12, 4.05]
masse_lodd = 2                                  # [kg]
k = fjaer_konst(tau, masse_lodd)                # [N/m]
print("Fjærkonstanten blir: %.3g [N/m]" % k)


# oppgave 5
def std_mean(arr):
    """
    Funksjon som finner usikkerheten i gjennomsnittet
    """
    N = len(arr)
    mean_arr = np.mean(arr)
    s_arr = np.sqrt((1 / (N - 1)) * np.sum((arr - mean_arr) ** 2))
    d_arr = s_arr / np.sqrt(N)
    return d_arr


d_tau = std_mean(tau)       # standardfeil i gjennomsnittet av periodetiden
d_k = std_mean(k)           # standardfeil i gjennomsnittet av fjærkonstanten

mean_tau = np.mean(tau)                                     # gjennomsnittet til periodetiden (s)
d_m = np.sqrt((2 * d_tau / mean_tau) ** 2) * masse_lodd     # usikkerhet i massen [g]
print("Masse usikkerhet (dm): %.2g [g]" % (d_m * 1000))


# oppgave 7
"""
Følsomhet målt med usikkerhet i sterkt dagslys med laser
"""
fol = (20.712 * 0.05 / 100 + 0.002) * 1000
print("Følsomheten ved sterkt lys utenfra: %.2g [mm]" % fol)

