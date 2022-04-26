import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

e = 1.6022e-19          # elektron ladning [eV]
h_c = 1.241e-6          # [eV * m]
c = 3e8                 # speed of light [m / s^2]
m_e = 511000            # [eV / c^2]

diameter = np.loadtxt("diameter.dat")


def lambda_min(u):
    energy = u                              # elektron energien gjennom anoden [eV]
    minimum = h_c / energy / 1e-12          # bølgelengde [m]
    print("Minste bølgelengden med %.f V over anoden: %.f pm" % (u, minimum))     # [picometer]


def vinkel_min(vinkel, fotoner):
    """
    Minste vinkel svarer til den maksimale energien i bremsestrålespekteret.
    Som vil si der intensiteten begynner å øke (se graf).
    """
    plt.plot(vinkel, fotoner)
    # ca. 18 grader
    # plt.show()


def rontgen_spenning(angle):
    to_d = 401e-12       # gitterkonstanten for LiF-krystall [m]
    lambda_min = to_d * np.sin(np.deg2rad(angle / 2))
    u = h_c / lambda_min
    print("Spenningen brukt for akselerasjon: %.f V" % u)


def korreksjonsfaktor(u):
    if len(u) > 1:
        u_x = u[0]
        u_y = u[1]
        f_x = 1 / np.sqrt(1 + (u_x / (2 * m_e)))
        f_y = 1 / np.sqrt(1 + (u_y / (2 * m_e)))
        f = np.array([f_x, f_y])
        print("Korreksjonsvektoren for %.f V: %.2f" % (u, f))

    else:
        f = 1 / np.sqrt(1 + (u[0] / (2 * m_e)))
        print("Korreksjonsfaktor f for %.f V: %.4f" % (u, f))


def vinkel_data(arr):
    n = len(arr)

    lambda_c = 2.426e-12
    u = arr[:, 0]
    d_ytre = arr[:, 1]

    lambda_i = lambda_c * np.sqrt(m_e / (2 * u))
    phi_i = lambda_i / d_ytre

    phi_avg = (1 / n) * sum(phi_i)
    phi_s = np.sqrt((1 / (n - 1)) * sum((phi_i - phi_avg) ** 2))
    delta_phi = phi_s / np.sqrt(n)
    print("Phi-verdien: %.3f +- %.3f" % (phi_avg / 1e-10, delta_phi / 1e-10))


# oppg 3 (korrekt)
lambda_min(20000)

# oppg 4 (korrekt)
to_nu = np.linspace(12, 25, 14)      # grader [2 * nu]
intensitet = np.array([130, 124, 133, 131, 128, 132, 138, 192, 244, 301, 348, 403, 462, 508])    # [fotoner per tid]
vinkel_min(to_nu, intensitet)

# oppg 5 (korrekt)
rontgen_spenning(18)

# oppg 7 (korrekt)
korreksjonsfaktor(np.array([8000]))

# oppg 8 (korrekt)
vinkel_data(diameter)

# oppg 9 (korrekt)
delta_d = np.sqrt((0.003 / 0.127) ** 2 + (0.08 / 8.31) ** 2)
print("Systematisk og empirisk usikkerhet for gitter i krystall: %.3f" % delta_d)

# oppg 10 (korrekt)
d10_d11 = (1 + np.cos(np.deg2rad(120 / 2))) / np.sin(np.deg2rad(120 / 2))
print("Forventet forhold mellom plykarbon-gitter: %.2f eller sqrt(3)" % d10_d11)
print("-----------------------------------------------------------------")


# LABDAGEN


