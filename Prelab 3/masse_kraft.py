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
    N = len(x)

    D = np.sum(x ** 2) - ((1 / N) * (np.sum(x) ** 2))
    E = np.sum(x * y) - ((1 / N) * np.sum(x) * np.sum(y))
    F = np.sum(y ** 2) - ((1 / N) * (np.sum(y) ** 2))

    alpha = E / D
    beta = np.mean(y) - (alpha * np.mean(x))

    d_alpha_2 = ((1 / (N - 2)) * (D * F - E ** 2 / D ** 2)) ** 2
    d_alpha = np.sqrt(d_alpha_2)
    return alpha, beta, d_alpha


print(best_straight_line(m, h))


# oppgave 4
m_2 = 2
tau = [4.12, 4.04, 4.16, 4.02, 4.03, 4.04, 3.89, 4.2, 4.12, 4.05]
N = len(tau)
k = np.zeros(N)
for i in range(N):
    k[i] = (m_2 / ((2 * np.pi * tau[i]) ** 2))

fj_konst = np.mean(k)


# oppgave 5
def std_mean(arr):
    mean_arr = np.mean(arr)
    s_arr = np.sqrt((1 / (N - 1)) * np.sum((arr - mean_arr) ** 2))
    d_arr = s_arr / np.sqrt(N)
    return d_arr


d_tau = std_mean(tau)
d_k = std_mean(k)

mean_tau = np.mean(tau)
d_m = np.sqrt((d_k / fj_konst) ** 2 + (2 * d_tau / mean_tau) ** 2)
print("Masse usikkerhet (dm): %.2g [g]" % (d_m * 1000))       # masse usikkerhet [g]


# oppgave 7
print(20.712 * 0.05 / 100 + 0.002)

