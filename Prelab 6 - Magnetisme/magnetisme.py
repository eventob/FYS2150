import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def susceptibility(f_z, cross_sec, b_1, b_2):
    """
    Funksjon for å beregne susceptibiliteten for vismut materialet.
    (Eksperiment 1)
    """
    mu_0 = 1.256e-6     # vacuum permeability [H/m]
    return - (2 * mu_0 * f_z) / (cross_sec * (b_1 ** 2 - b_2 ** 2))


def avmagnetiseringsfaktor(a_para, a_orto):
    """
    Funksjon som regner ut avmagnetiseringsfaktoren for legemet i magnetfeltet.
    (Eksperiment 2)
    """
    f = a_para / a_orto                     # akselengder forhold [enhetsløs]
    epsilon = np.sqrt(1 - (1/(f ** 2)))     # eksentrisitet [enhetsløs]

    d_para = (1 - (1 / (epsilon ** 2))) * (1 - ((1 / (2 * epsilon)) * np.log((1 + epsilon) / (1 - epsilon))))
    d_orto = (1 - d_para) / 2
    return d_para, d_orto


def magnetfeltet_h0(i):
    """
    Funksjon for å finne det induserte magnetfeltet i primærspolen
    (Eksperiment 3)
    """
    return N * i / L


def endring_magnetiskflukstetthet(s_1, s_2, i_1, i_2, cross_sec, viklinger):
    """
    Funksjon som regner ut magnetfeltet for objektet
    (Eksperiment 3)
    """
    k = 1               # kalibreringsfaktor oppgitt av instrumentet
    D = 1               # dempningsfaktor, innstilles på integratoren
    kappa = k * D

    delta_B_I = kappa * (s_2 - s_1) / (viklinger * cross_sec)       # endring i flukstetthet [T]
    B_I = delta_B_I / 2
    I = (i_1 + i_2) / 2     # strøm fra hver av ytterpunktene
    return B_I, I