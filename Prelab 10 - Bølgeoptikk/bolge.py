import numpy as np
import matplotlib.pyplot as plt


def kvanta_lambda(R, n_2):
    lambda_i = (1 / R) * (1 / (((0.5 ** 2)) - (1 / (n_2 ** 2))))
    return lambda_i


def diffrak_illuminans(x, R, a, N=1, A=1):
    """
    x er avstanden til spalten fra en avstand x i mønsteret, c er en konstant, R er avstanden fra spalten til
     midten av skjermen med inteferensmønsteret, a er spaltebredden, N er antall spalter, A er avstanden
     mellom flere splater (konstant).
    """
    lambda_hene = 632.8e-9          # [m]
    c = np.pi / (lambda_hene * R)      # konstant
    if N == 1:
        e_1 = (1/N) * (np.sin(a * c * x) / (a * c * x)) ** 2
        plt.plot(x * 1e3, e_1)
        plt.grid(), plt.xlabel("Avstand (mm)"), plt.ylabel("Illuminans")
        plt.show()
    else:
        e_n = (np.sinc(a * c * x) / (a * c * x)) ** 2 * (np.sinc(N * A * c * x) / np.sinc(A * c * x)) ** 2
        print(e_n)


def b_felt(d_1, d_2, d_3):
    t = 0.003               # [m]
    hc = 1.98644568e-25     # [Jm]
    mu_b = 9.274009994e-24  # [J/T]

    delta = ((d_2 ** 2 - d_1 ** 2) / (d_3 ** 2 - d_1 ** 2))
    b = (hc / (4 * mu_b * t)) * delta
    print(b * 1e3, delta)

"""
# oppg 2 (korrekt)
kvanta_lambda(1.097e7, 6, 2)        # [nm]

# oppg 5 (korrekt)
diffrak_illuminans(np.linspace(-0.05, 0.05, 1000), 5, 0.00012)

# oppg 10 (korrekt-ish)
b_felt(0.0124, 0.0134, 0.0152)      # [mT]
"""

# LABDAGEN
# DEL A
print('-----------------------------------------------------')
def spalte_bredde(n, R, x):
    lambda_i = 632.8e-9     # [m]
    return n * lambda_i * R / x


# diffrak_illuminans()


def teori_diameter(a, lambda_i, R):
    omega_0 = 3.832, 7.016, 10.173, 13.324
    r = np.pi * a * omega_0 / (lambda_i * R)
    print(2 * r)


# DEL B
print('-----------------------------------------------------')
dist = 0.0254 / 30000      # [m]
print(dist)
m = 1

alpha_nu = np.array([149.4, 147.3, 145.2, 129])
alpha_h = np.array([210.9, 213, 215, 230.5])
delta_alpha = 0.1 * (np.pi / 180)
delta_theta = 0.5 * np.sqrt(delta_alpha ** 2 + delta_alpha ** 2)


theta = (abs(alpha_h - alpha_nu) / 2)
delta_lambda = np.sqrt(delta_theta * theta / np.tan(theta)) * theta
lambda_calc = dist * np.sin(np.deg2rad(theta)) / m
print(np.flip(theta), delta_theta * theta)
print(kvanta_lambda(1.097e7, np.array([3, 4, 5, 6, 7])) / 1e-9)
print(np.flip(lambda_calc) / 1e-9, delta_lambda)

alpha_nu_he = np.array([302.8, 311, 319.5, 321.4, 323.3])
alpha_h_he = np.array([46.8, 39, 30.7, 28.9, 27.1])

theta_he = (abs(alpha_nu_he - alpha_h_he)) / 2
lambda_he_calc = dist * np.sin(np.deg2rad(theta_he)) / m

print(np.flip(theta_he), delta_theta * theta)
print(lambda_he_calc / 1e-9, delta_lambda)

# DEL C
print('-----------------------------------------------------')
h = 6.626e-34

mag3 = np.array([520, 533])
d3min = np.array([371, 450, 600])
d3plus = np.array([397, 472, 619])

mag4 = np.array([677, 694])
d4min = np.array([357, 460, 593])
d4plus = np.array([385, 484, 613])


def eksp_mub(dmin, dplus, mag_field):
    t = 0.003       # [m]
    hc = 1.98644568e-25  # [Jm]
    d_1, d_2, d_3 = abs(dmin[0] + dplus[0]) / 2, abs(dmin[1] + dplus[1]) / 2, abs(dmin[2] + dplus[2]) / 2
    delta = (d_2 ** 2 - d_1 ** 2) / (d_3 ** 2 - d_1 ** 2)

    mu_b_pm = hc / (4 * t) * (delta / (mag_field / 1e3))
    avg_mu_b = 0.5 * (mu_b_pm[0] + mu_b_pm[1])
    return avg_mu_b


mu_b3 = eksp_mub(d3min, d3plus, mag3)
mu_b4 = eksp_mub(d4min, d4plus, mag4)

final_mub = (mu_b3 + mu_b4) / 2
delta_nu3 = 2 * final_mub * (mag3 / 1e3) / h
delta_nu4 = 2 * final_mub * (mag4 / 1e3) / h

delta_mub = ((((9.34729 - 8.98937) / 2) + ((9.34729 - 9.01911) / 2))) / 1e24
print(final_mub / 1e-24, delta_mub / 1e-24, 3e8 / 643.8e-9, delta_nu3, delta_nu4)        # [J/T]


