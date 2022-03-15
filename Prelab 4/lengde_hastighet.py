import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import probplot
from scipy.stats import linregress

print(0.766 / 0.065)
print(15.83 / 0.000017)
du_1 = (0.766 * (0.02 / 100) + 0.006)
du_2 = (15.83 * (0.02 / 100) + 0.002)
print(du_1, du_2)

# oppgave 3
def sound_speed(temp, fm_f=0.999):
    c = 331.1 + (0.606 * temp)
    return c * (1 - (1 / fm_f))

di_1 = (0.065 * (0.05 / 100) + 0.002)
di_2 = (0.000017 * (0.05 / 100) + 0.000003)
print(di_1, di_2)

dr = np.sqrt((du_1 / 0.766) ** 2 + (di_1 / 0.065) ** 2) * 11.78
dr_2 = np.sqrt((du_2 / 15.83) ** 2 + (di_2 / 0.000017) ** 2) * 931200
print(dr, dr_2)

print(sound_speed(25))

# oppgave 4
dv_v = 1 / 10   # hastighetsoppløsning
s_freq = 10     # [kHz]
c = 343.12      # lydhastighet [m/s]


def hast_tid_maaling(freq, dv, N):
    d_f = freq * ((c / (c - dv)) - 1) * 1000
    t = N * (1 / d_f)
    dist = 1 * t
    return d_f, t, dist


print(hast_tid_maaling(s_freq, dv_v, 5))


