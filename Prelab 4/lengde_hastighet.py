import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import probplot
from scipy.stats import linregress


# oppgave 3
def sound_speed(temp, fm_f=0.999):
    c = 331.1 + (0.606 * temp)
    return c * (1 - (1 / fm_f))


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


