import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def brytningsindeks(theta_i):
    return np.sin(theta_i) / np.sin((180 - (theta_i + 90)))


def prelab():
    data_1 = np.loadtxt("polarisering1.dat")
    data_2 = np.loadtxt("polarisering2.dat")
    data_3 = np.loadtxt("polarisering3.dat")

    plt.plot(data_1[0], data_1[1], label="Datasett 1")
    plt.plot(data_2[0], data_2[1], label="Datasett 2")
    plt.plot(data_3[0], data_3[1], label="Datasett 3")
    plt.legend(), plt.grid()
    plt.xlabel("Elektrisk vektor ($E_y$)"), plt.ylabel("Elektrisk vektor ($E_z$)")
    plt.show()


def eksp_2(e_array, i_0, yes=1):
    theta = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    x = np.cos(theta * 0.01745) ** 2
    y_data = e_array
    y_teori = i_0 * x

    line = linregress(x, y_data)
    data_stig, data_konst = line.slope, line.intercept
    approx = (data_stig * x) + data_konst


    plt.plot(x, y_data, 'ro', label="Målinger")
    if yes == 0:
        plt.plot(x, approx, label="Lineær tilpasning")
        plt.plot(x, y_teori, label="Teoretisk Malus")
    elif yes == 2:
        y_teori_2 = np.array([101, 97, 88, 74, 59, 41, 26, 14, 6, 4])[1:-1] * np.cos(theta[1:-1]) ** 2
        plt.plot(x[1:-1], y_teori_2, label="Teoretisk Malus")
    plt.xlabel("Transmisjonsretning [$\cos^2\\theta$]"), plt.ylabel("Illuminans E [LUX]")
    plt.grid(), plt.legend()



e_array_1 = np.array([101, 97, 88, 74, 59, 41, 26, 14, 6, 4])
e_array_2 = np.array([2, 5, 9, 14, 17, 16, 12, 7, 3, 2])
eksp_2(e_array_1, 100, 0)
plt.savefig("oppg_2.1.png")
plt.show()
eksp_2(e_array_2, 100, 2)
plt.savefig("oppg_2.2.png")
plt.show()


def eksp_3(phi, intensity):
    plt.plot(phi, intensity / 100, 'go', label="Intensitet data")
    plt.xlabel("Innfallsvinkel $\\phi$ [deg]"), plt.ylabel("Intensitet [%]")


intensity_1 = np.array([])      # intensitet s-polarisert
phi_1 = np.array([])            # vinkler for s-polarisering mellom 0 og 90 grader
intensity_2 = np.array([])      # intensitet p-polarisert
phi_2 = np.array([])            # vinkler for p-polarisering rundt lavest intensitet

eksp_3(phi_1, intensity_1)
eksp_3(phi_2, intensity_2)
plt.grid(), plt.legend()
# plt.show()


def indeks_prisme(phi_p):
    n_1 = 1.0
    n_2 = n_1 * np.tan(phi_p)
    return n_2


# print(indeks_prisme(min(phi_2)))

