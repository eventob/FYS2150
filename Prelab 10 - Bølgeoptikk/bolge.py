import numpy as np
import matplotlib.pyplot as plt


def kvanta_lambda(R, n_2, n_1):
    lambda_i = (1 / R) * (1 / ((1 / (n_1 ** 2)) - (1 / (n_2 ** 2))))
    print(lambda_i * 1e9)


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


# oppg 2 (korrekt)
kvanta_lambda(1.097e7, 6, 2)        # [nm]

# oppg 5 (korrekt)
diffrak_illuminans(np.linspace(-0.05, 0.05, 1000), 5, 0.00012)

# oppg 10 (korrekt-ish)
b_felt(0.0124, 0.0134, 0.0152)      # [mT]


# LABDAGEN
# DEL A
def spalte_bredde(n, R, x):
    lambda_i = 632.8e-9     # [m]
    return n * lambda_i * R / x


diffrak_illuminans()


def teori_diameter(a, lambda_i, R):
    omega_0 = 3.832, 7.016, 10.173, 13.324
    r = np.pi * a * omega_0 / (lambda_i * R)
    print(2 * r)


# DEL B
kvanta_lambda(1.097e-7, np.array([3, 4, 5, 6, 7]), 0.5)


# DEL C

