import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

poisson = np.loadtxt("poisson.csv")
spektrum = np.loadtxt("spektrum.csv")


def std_dev(arr):
    """
    Funksjon som finner standardavviket til gjennomsnittet
    """
    N = len(arr)
    avg_k = (1/N) * np.sum(arr)
    dev_std = np.sqrt((1 / (N - 1)) * np.sum((arr - avg_k) ** 2))
    print("Standard avviket til gjennomsnittet: %g" % dev_std)


def gm_eff(n_r, n_b, A, r, d):
    """
    Funksjon som regner effektiviteten til en gm-måler.
    """
    omega = np.pi * r ** 2 / d ** 2
    gm = ((n_r - n_b) / (A * (omega / (4 * np.pi)))) * 1e2      # [%]
    print('GM_eff: %g' % gm)


def lambert_lov(skj_arr, n_arr):
    """
    (Ufullstendig) Funksjon som anslår svekkingskoeffisient og dens statistiske usikkerhet.
    """
    line = linregress(skj_arr, n_arr)
    y_line = (line.slope * skj_arr) + line.intercept
    plt.plot(skj_arr, y_line)
    plt.plot(skj_arr, n_arr)
    plt.grid()
    plt.xlabel("Tykkelse [m]"), plt.ylabel("Tellinger [$s^{-1}$]")
    plt.savefig("mu_tykkelse.png")
    plt.show()
    mu, delta_mu = abs(line.slope) / 10, line.stderr / 10
    print("Mu: %g, delta_Mu: %g" % (mu, delta_mu))


def gamma_energi(i_arr, e_arr):
    """
    Funksjon som finner dispersjonen (delta_E) i hver kanal og nullpunktsenergien for et gammaspektrometer
    """
    line = linregress(i_arr, e_arr)
    return line.slope, line.intercept


def linear_fwhm(spek, e_0, delta_e):
    buff = 2
    kanal = np.linspace(0, len(spek), len(spek))
    fwhm_y = np.max(spektrum) / 2 + (abs(e_0) / delta_e)     # half maximum, fjernet bakgrunnsstråling
    fwhm_x = kanal[np.logical_and(spek < fwhm_y + buff, spek > fwhm_y - buff)]
    if len(fwhm_x) != buff:
        raise Exception("No x-values within the defined buffer area!")
    fwhm = (fwhm_x[1] - fwhm_x[0]) * delta_e
    print("FWHM keV: %g" % fwhm)    # feil (243.3), feil (243), korrekt (264)

    plt.plot(kanal, spektrum)
    plt.axhline(fwhm_y, color='red', linestyle='--')
    plt.axhline(np.max(spektrum), color='orange', linestyle='dotted')
    plt.axvline(fwhm_x[1], 0, fwhm_y / np.max(spektrum), color='red', linestyle='--')
    plt.axvline(fwhm_x[0], 0, fwhm_y / np.max(spektrum), color='red', linestyle='--')
    plt.xlabel("Kanal"), plt.ylabel("Tellinger/kanal [n / kanal]")
    plt.grid()
    plt.show()


# PRE-LAB OPPGAVER
# spm 2, 3, 4
print("Gjennomsnitt poisson: %g" % np.mean(poisson))     # korrekt
std_dev(poisson)     # korrekt
gm_eff(23, 2, 1e6, 0.02, 0.2)     # korrekt

# spm 5, 6
skj_x = np.array([0, 5, 10, 15, 20, 25]) / 1e3           # [m]
tell_n = 1000 / (np.array([32.89, 54.84, 94.52, 149.49, 272.60, 427.33])) - 0.248   # [s^-1]
lambert_lov(skj_x, tell_n)  # feil (-275.9), feil (275.9), feil (0.276), feil (13.36), feil (13.35), feil (0.013)

# spm 7
z = abs(np.log(0.01) / 107.4) * 1e3
print("Tykkelse z: %g" % z)     # korrekt

# spm 8
delta_z = (0.04 * z)
print("delta_z: %g" % delta_z)      # feil (1 / 0.08), korrekt (0.04 * z)

# spm 9, 10
i = np.array([410, 773])        # kanalen til centroiden av toppen
e = np.array([662, 1275])       # energien til gammastrålingen gitt fra desintegrasjonsskjema
print("stig: %g, skjer: %g" % gamma_energi(i, e))   # feil (1.7, 2), korrekt (1.69), korrekt (-30)


# spm 11
# linear_fwhm(spektrum, -35, 2)

# LABDAGEN
def data_behandling(data):
    ch, mal = data[:, 0], data[:, 1]
    data_max = ch[np.where(mal == np.max(mal))][0]
    # if len(data_max) < 1:
    est_min = int(data_max - 50)
    est_max = int(data_max + 50)
    fit = np.polyfit(ch[est_min:est_max], mal[est_min:est_max], 50)
    est = np.poly1d(fit)
    fit_top = ch[np.where(est(ch[est_min:est_max]) == np.max(est(ch[est_min:est_max])))]
    plt.plot(ch, mal)
    plt.plot(ch[est_min:est_max], est(ch[est_min:est_max]))
    # plt.axvline((est_min + fit_top), color='red', linestyle='dotted')
    plt.grid()
    plt.show()


cs_137 = np.loadtxt("Cs137_spektrum.asc")
co_60 = np.loadtxt("Co60_spektrum.asc")
data_behandling(cs_137)
data_behandling(co_60)

i_data = np.array([396, 791])
e_data = np.array([662, 1332])

print(gamma_energi(i_data, e_data))
print(694 * 1.6962025)
ra_topp = np.array([841, 663, 370, 214, 180, 149, 106, 47])
print(ra_topp * 1.6962025)
print(1024 * 1.6962)
