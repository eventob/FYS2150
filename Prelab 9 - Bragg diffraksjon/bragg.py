import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

diameter = np.loadtxt("diameter.dat")


#def lambda_min():
    #?????


def vinkel_min():
    """
    Minste vinkel svarer til den maksimale energien i bremsestrålespekteret.
    """
    array = np.array([130, 124, 133, 131, 128, 132, 138, 192, 244, 301, 348, 403, 462, 508])
    vinkel = np.linspace(12, 25, 14)
    print(vinkel[np.where(array == max(array))[0]])


def rontgen_spenning():
    e = 1.602e-19       # [C]
    return 508 / e


def korreksjonsfaktor(U):
    """
    https://virtuelle-experimente.de/en/kanone/relativistisch/relativistisch.php
    """
    c = 3e8
    e = 1.602e-19       # [C]
    m_e = 9.109e-31
    f = 1 - (U * e / (m_e * c ** 2))
    print(f)


def vinkel_data():
    aks_u = diameter[:, 0]
    ytre_d = diameter[:, 1]

    phi_avg = sum(lambda_i / ytre_d)


vinkel_min()
print(rontgen_spenning())
korreksjonsfaktor(8000)
vinkel_data()
