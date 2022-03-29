import numpy as np
import matplotlib.pyplot as plt


def brytningsindeks(theta_i):
    return np.sin(theta_i) / np.sin((180 - (theta_i + 90)))


data_1 = np.loadtxt("polarisering1.dat")
data_2 = np.loadtxt("polarisering2.dat")
data_3 = np.loadtxt("polarisering3.dat")

plt.plot(data_1[0], data_1[1], label="Datasett 1")
plt.plot(data_2[0], data_2[1], label="Datasett 2")
plt.plot(data_3[0], data_3[1], label="Datasett 3")
plt.legend(), plt.grid()
plt.xlabel("Elektrisk vektor ($E_y$)"), plt.ylabel("Elektrisk vektor ($E_z$)")
plt.show()
