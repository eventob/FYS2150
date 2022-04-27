import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

e = 1.6022e-19          # elektron ladning [eV]
h_c = 1.241e-6          # [eV * m]
c = 3e8                 # speed of light [m / s^2]
m_e = 511000            # [eV / c^2]

diameter = np.loadtxt("diameter.dat")


def lambda_min(u):
    """
    Minste bølgelengde til røntgenstråling ved hjelp av anodespenning.
    """
    energy = u                              # elektron energien gjennom anoden [eV]
    minimum = h_c / energy / 1e-12          # bølgelengde [m]
    return minimum
    # print("Minste bølgelengden med %.f V over anoden: %.f pm" % (u, minimum))     # [picometer]


def min_vinkel_2v(vinkel, fotoner):
    """
    Minimumsvinkelen til røntgenstråling ved hjelp av LiF-krystall.
    Som vil si der intensiteten begynner å øke (se graf).
    """
    plt.plot(vinkel, fotoner, 'o', label="Data",)
    plt.xlabel("Vinkel $2\cdot \\theta$ (deg)"), plt.ylabel("Intensitet (fotoner)")
    plt.title("Røntgenspektrum for LiF-krystall")
    plt.grid()
    # ca. 18 grader


def rontgenror_spenning(angle):
    """
    Spenning over røntgenrøret ved bruk av en LiF-krystall og minste vinkelen (2 * v)
    """
    to_d = 401e-12       # gitterkonstanten for LiF-krystall [m]
    lambda_min = to_d * np.sin(np.deg2rad(angle / 2))
    u = h_c / lambda_min
    print("Spenningen brukt for akselerasjon: %.f V" % u)


def korreksjonsfaktor(u):
    """
    Funksjonen gir korreksjonsfaktoren for en gitt spenning U.
    """
    if len(u) > 1:
        f = np.zeros(len(u))
        for i in range(len(u)):
            f[i] = 1 / np.sqrt(1 + (u[i] / (2 * m_e)))

        print(f)
        # print("Korreksjonsvektoren for %.f V: %.2f" % (u, f))

    else:
        f = 1 / np.sqrt(1 + (u[0] / (2 * m_e)))
        print("Korreksjonsfaktor f for %.f V: %.4f" % (u, f))


def vinkel_data(spenning, ytre):
    """
    Funksjon som regner ut phi, gitt en array med akselerasjonsspenninger og diameteren til
    den ytre ringen (se laboppgave C2)
    """
    n = len(spenning)

    lambda_c = 2.426e-12
    u = spenning
    d_ytre = ytre

    lambda_i = lambda_c * np.sqrt(m_e / (2 * u))
    phi_i = lambda_i / d_ytre

    phi_avg = (1 / n) * sum(phi_i)
    phi_s = np.sqrt((1 / (n - 1)) * sum((phi_i - phi_avg) ** 2))
    delta_phi = phi_s / np.sqrt(n)
    return phi_avg, delta_phi, lambda_i / 1e-12, phi_i
    # print("Phi-verdien: %.3f +- %.3f" % (phi_avg / 1e-10, delta_phi / 1e-10))


# oppg 3 (korrekt)
lambda_min(20000)

# oppg 4 (korrekt)
to_nu = np.linspace(12, 25, 14)      # grader [2 * nu]
intensitet = np.array([130, 124, 133, 131, 128, 132, 138, 192, 244, 301, 348, 403, 462, 508])    # [fotoner per tid]
# min_vinkel_2v(to_nu, intensitet)
# plt.show()

# oppg 5 (korrekt)
rontgenror_spenning(18)

# oppg 7 (korrekt)
korreksjonsfaktor(np.array([8000]))

# oppg 8 (korrekt)
# vinkel_data(diameter)

# oppg 9 (korrekt)
delta_d = np.sqrt((0.003 / 0.127) ** 2 + (0.08 / 8.31) ** 2)
print("Systematisk og empirisk usikkerhet for gitter i krystall: %.3f" % delta_d)

# oppg 10 (korrekt)
d10_d11 = (1 + np.cos(np.deg2rad(120 / 2))) / np.sin(np.deg2rad(120 / 2))
print("Forventet forhold mellom plykarbon-gitter: %.2f eller sqrt(3)" % d10_d11)
print("A----------------------------------------------------------------")


# LABDAGEN
# A
theta = np.linspace(12, 22, 21)
lab_fotoner = np.array([100, 111, 113, 120, 128, 134, 167, 186, 230, 278, 398, 468, 547, 576, 601, 673, 740, 800, 849, 917, 955])
best_angle = 16.5
rontgenror_spenning(best_angle)

"""
min_vinkel_2v(theta, lab_fotoner)
plt.axvline(best_angle, color='red', linestyle='dotted', label=best_angle)
plt.legend()
plt.show()
"""

# B
print("B----------------------------------------------------------------")
kobber_d = 629 * 1e-12      # KCl gitteravstand
n = np.array([1, 2, 1, 2])
alpha_lambda = (154.4 + 154) / 2
beta_lambda = (139.2 + 138.1) / 2
lambda_list = np.array([alpha_lambda, alpha_lambda, beta_lambda, beta_lambda]) * 1e-12      # [m]
ba_vink = 2 * np.rad2deg(np.arcsin(n * lambda_list / kobber_d))

print(alpha_lambda, beta_lambda)
print(ba_vink)

en_orden = np.linspace(23, 30, 15)
to_orden = np.linspace(51, 60, 19)
en_intens = np.array([229, 188, 305, 529, 748, 755, 515, 421, 946, 2015, 2586, 2198, 976, 136, 139])
to_intens = np.array([70, 127, 173, 186, 94, 48, 47, 46, 46, 44, 44, 59, 89, 203, 542, 556, 382, 141, 53])

"""
plt.plot(en_orden, en_intens, 'o', label="Første orden")
plt.axvline(28, color='green', linestyle='dotted', label='alphatopp: %g' % 28)
plt.axvline(25.5, color='red', linestyle='dotted', label='betatopp: %g' % 25.5)
plt.xlabel("Vinkel $2\cdot\\theta$ (deg)"), plt.ylabel("Intensitet")
plt.grid(), plt.legend(), plt.title("Røntgenspektrum for KCL-krystall")
plt.show()

plt.plot(to_orden, to_intens, 'o', label="Andre orden")
plt.axvline(58.5, color='green', linestyle='dotted', label='alphatopp: %g' % 58.5)
plt.axvline(52.5, color='red', linestyle='dotted', label='betatopp: %g' % 52.5)
plt.xlabel("Vinkel $2\cdot\\theta$ (deg)"), plt.ylabel("Intensitet")
plt.grid(), plt.legend(), plt.title("Røntgenspektrum for KCL-krystall")
plt.show()
"""

theta_orden = np.array([28, 58.5, 25.5, 52.5]) / 2
kobber_gitter = n * lambda_list / np.sin(np.deg2rad(theta_orden)) / 1e-12       # [pm]
print(kobber_gitter)

# C
print("C----------------------------------------------------------------")
r = 10e6    # motstand 10 M ohm
volt_nom = np.linspace(3, 5, 11) * 1e3
strom = np.array([0.027, 0.029, 0.031, 0.032, 0.034, 0.036, 0.038, 0.040, 0.041, 0.043, 0.045]) / 1e3
volt = volt_nom - (r * strom)

d_in_1 = np.array([27.3, 26.4, 26, 22.7, 23.9, 22.9, 22, 21.6, 22.3, 21.3, 20])
d_in_2 = np.array([47.2, 48.3, 47.8, 44, 43.4, 43, 40.5, 40.4, 39.1, 39, 37.3])
d_out_1 = np.array([31.6, 29.8, 28.9, 27.3, 27.3, 27, 26.9, 25.4, 24.7, 25.4, 22.5])
d_out_2 = np.array([55.2, 52, 52.3, 51, 48.9, 47.6, 46.4, 45.2, 44.8, 41.9, 42.2])

d_avg_inner = ((d_in_1 + d_out_1) / 2) / 1e3
d_avg_outer = ((d_in_2 + d_out_2) / 2) / 1e3

print(d_avg_inner, d_avg_outer)

print(volt)
print(vinkel_data(volt, d_avg_inner))
print(vinkel_data(volt, d_avg_outer))

korreksjonsfaktor(np.array([1000, 5000, 20000, 50000, 100000]))
print(2 * 0.125 * 8.17e-10 / 1e-12)
print(np.sqrt((0.002 / 0.125) ** 2 + (0.06 / 8.17) ** 2) * 8.17e-10 / 1e-12)
print(2 * 0.125 * 4.53e-10 / 1e-12)
print(np.sqrt((0.002 / 0.125) ** 2 + (0.02 / 4.53) ** 2) * 4.53e-10 / 1e-12)

print(204.25 / 113.25)
