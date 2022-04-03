import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = np.genfromtxt("faraday.csv", delimiter=',')
data = data[1:]
mag_flux = data[:, 0]
angle = data[:, 1]


# Funksjoner for lab-utregninger
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


def verdet_konstant(b_arr, angle_arr, length, d_length):
    line = linregress(b_arr, angle_arr)
    d_v = line.slope
    v = d_v / length
    print(v, d_v)


# verdet_konstant(mag_flux, angle, 0.03, 0.001)

m = np.array([0, 0, 0, 0.02, 0.04, 0.07, 0.1, 0.14, 0.17, 0.21, 0.23, 0.27, 0.29])
b_1 = np.array([0.2, 0.8, 1.4, 2.1, 2.8, 3.4, 4.0, 4.5, 5.0, 5.5, 5.9, 6.2, 6.5])
b_2 = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5])
b_12 = sum(b_1) / 13
b_22 = sum(b_2) / 13
m_avg = sum(m) / 13

b = ((b_12 ** 2 - b_22 ** 2) / 1e3)     # [T]
# print(b, m_avg)
f_z = (m_avg / 1e4) * 9.81
# print(f_z)

alp = f_z / b
# print(alp)

# print(np.sqrt((0.000002 / 0.00052) ** 2 + (0.00001 / 0.0084) ** 2))
mu_0 = 4 * np.pi * 1e-7
# print(- 2 * mu_0 * alp / ((10.45 / (2 * 1e3)) ** 2 * np.pi / 1e4))

"""
NOTES: Det er to spørsmål på prelabben som er feil og bør settes inn i dette programmet
før labben i morgen 16.03!
"""

def avmagfakt(a_para, a_orto):
    f = a_para / a_orto
    epsilon = np.sqrt(1 - (1 / (f ** 2)))
    d_para = (1 - (1 / (epsilon ** 2))) * (1 - ((1 /(2 * epsilon)) * np.log((1 + epsilon)/(1 - epsilon))))
    d_orto = (1 - d_para) / 2
    return d_para, d_orto


a_p = np.array([64.54, 59.87, 63.45, 207])
a_o = np.array([9.93, 6.87, 63.45, 8.61])

for i in range(len(a_p)):
    print(avmagfakt(a_p[i], a_o[i]))

print((4 * np.pi * 1e-7) * 244 * 5 / (275 + 64.54) * 1e3)
print(5.74 * 4 * np.pi * 1e-7)


print(np.pi * (3.25 ** 2))

def magnet_fluk(s_diff):
    k = 0.98
    d = 10
    n = 130
    A = 3.32 * 1e-5
    return k * d * s_diff / (2 * n * A)


s = np.array([0.07, 0.32, 0.53, 0.82, 1.06, 1.16, 1.23, 1.28])
I_eksp3 = np.array([0.055, 0.205, 0.285, 0.375, 0.475, 0.545, 0.605, 0.675])
maggi = magnet_fluk(s) / 1e3
h_0 = 344 * I_eksp3 / 0.315
m_0 = maggi / (4 * np.pi * 1e-7) - h_0
print(maggi)
print(h_0)
print(m_0 / 1e3)
plt.plot(I_eksp3, maggi, color='b')
plt.plot(I_eksp3, maggi, 'ro')
plt.xlabel("Strøm [I]"), plt.ylabel("Magnetisk flukstetthet [B]")
plt.savefig("mag_1.png")
plt.show()

plt.plot(h_0, m_0 / 1e3, color='b')
plt.plot(h_0, m_0 / 1e3, 'ro')
plt.xlabel("Magnetisk feltstyrke [$H_0$]"), plt.ylabel("Magnetisering [$M_0 \\cdot 10^3$]")
plt.savefig("mag_2.png")
plt.show()

