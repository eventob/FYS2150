import numpy as np
import matplotlib.pyplot as plt

"""
print((11.5 ** 2 + 0.1 ** 2) / (2 * 0.1))

R = 280 / 1e3
n = 1.520
print(abs((n - 1) * ((-1/R) - (1/R)) * 1e3))

"""
def r(x, d):
    return (x ** 2 + d ** 2) / (2 * d)


# LABDAGEN
# Eksp 1
def linseformel(s, s_mk):
    f_over = (1 / s) + (1 / s_mk)
    return 1 / f_over


print(linseformel(24.9, 115.1) * 1e1)       # [mm]
print(linseformel(24.6, 115.4) * 1e1)       # [mm]
print(r(12, 0.37))
print(194.78 / (2 * (1.520 - 1)))
print(0.01 * 194.78 / 0.37)


# Eksp 5
middel_d = np.array([185.6, 112.4, 73.9, 17, 10.3, 9.1, 8.9])
var_d = np.array([20.41, 8.56, 5.72, 3.18, 1.93, 1.38, 1.06]) / 2
print(var_d / middel_d)

plt.plot(middel_d, var_d)
plt.xlabel("Middelverdi D"), plt.ylabel("Varians for en partikkel Var(D)"), plt.grid()
plt.savefig("varians_partikkel.png")
plt.show()

signal_stoy = np.sqrt(var_d)
plt.plot(middel_d, signal_stoy)
plt.xlabel("Middelverdi D"), plt.ylabel("Signal/Støy"), plt.grid()
plt.savefig("signal_stoy.png")
plt.show()


