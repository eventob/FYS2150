import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def terminal_rayleigh(d):
    """
    ????
    (Turbulent fall)
    Funksjonen regner ut terminalhastigheten til objektet ved Rayleigh strømning.
    """
    # c_r = f_d / (rho * r ** 2 * vel ** 2)
    return d ** (1 / 2)


def terminal_stokes(rho_kule, rho_medium, mu_medium, diameter):
    """
    (Laminært fall)
    Funksjonen regner ut terminalhastigheten til objektet ved Stokes strømning.
    """
    g = 9.81  # gravitasjonskonstanten
    gamma = 1 / 18  # konstant for en kule fra Stokes' lov
    C_s = 6 * np.pi  # stokes-koeffisienten av en kule
    nu_s = mu_medium / rho_medium  # dynamsiske viskositet over tetthet til mediet
    g_hat = ((rho_kule / rho_medium) - 1) * gamma * g

    # returnerer mm / s
    return ((g_hat / nu_s) * (diameter ** 2)) * 1000


def terminal_plot(arr_data, arr_label):
    for i in range(1, 4):
        plt.plot(arr_data[0], arr_data[i], '-o', label=[arr_label[i], (i + 1)])
    plt.xscale('log'), plt.yscale('log')
    plt.xlabel("Radius [$log (m)$]"), plt.ylabel("Hastighet [$log (m/s)$]"), plt.title("Terminal hastighet for kuler")
    plt.legend(), plt.grid('log')
    plt.show()


def linear_regression(x_arr, y_arr, label):
    """
    Funksjon som finner stigningstall ved lineær regresjon og returnerer korrekt verdi.
    """
    line = linregress(x_arr, y_arr)
    stigningstall = line.slope
    konstantledd = line.intercept
    standard_feil = line.stderr
    print("Gjennomsnittshastighet for %s strømning: %g +- %g [mm/s]" % (label, stigningstall, standard_feil))
    print("R i andre: %.3g" % line.rvalue ** 2)
    return stigningstall, standard_feil, konstantledd


def terminal_video(frames, length, fps=60):
    """
    Funksjon for å beregne terminalhastighet fra videoer.
    """
    terminal = np.zeros(len(frames))
    for i in range(len(frames)):
        terminal[i] = length * (fps / frames[i])
        # print("Terminalhastighet fra video: %.5g [mm/s]" % terminal[i])

    return terminal


def linear_graph(x_arr, slope, const):
    return slope * x_arr + const


def stokes_coefficient(mu_medium, rho_medium, rho_kule, b_s, g=9.81):
    b_s = np.log(b_s)
    nu_s = mu_medium / rho_medium  # dynamsiske viskositet over tetthet til mediet
    g_hat = nu_s * np.exp(b_s)
    gamma = g_hat / (((rho_kule / rho_medium) - 1) * g)

    return np.pi / (3 * gamma) / 1e3


def plot_sammenligning(x_arr, y_arr, where):
    plt.plot(x_arr * 1e3, y_arr, 'o', label=where)
    plt.title('Terminalhastighet isopor- og stålkuler')
    plt.legend()
    plt.xlabel('Diameter [mm]'), plt.ylabel('Hasitghet [mm/s]')
    # plt.grid(which='both')
    # plt.savefig('terminal_hastigheter.png')


def plot_analyse():
    plt.xlim(min(d_array) * 1e3, max(d_array) * 1e3), plt.ylim(8, 1e4)
    plt.xscale('log'), plt.yscale('log')
    plt.xlabel('Diameter [mm]'), plt.ylabel('Terminal hastighet ($v_T$) [mm/s]')
    plt.grid(which='both')
    plt.legend()


rho_kuler = [21.1, 1042, 7905]  # tetthet for (isopor, plast, stål) [kg / m^3]
rho_medium = [1.225, 889, 998]  # tetthet mediumer (luft, olje, vann) [kg / m^3]
nu_medium = [15.2 * 1e-6, 214 * 1e-6, 1.0 * 1e-6]  # viskositet mediumer (luft, olje, vann) [10^-6 m^2 / s]
mu_medium = [1.862 * 1e-5, 0.190, 0.998 * 1e-3]  # dynamisk viskositet (luft, olje, vann) [kg / ms]

d_array = np.linspace(0.0005, 0.15, 10000)  # 1 mm til 200 mm [m]
d_liten = np.array([1, 2, 3, 5, 8, 10]) / 1000  # diameter for små stålkuler i olje, 1 mm til 10 mm [m]
d_stor = np.array([10, 15, 20, 27, 30, 38]) / 1000  # diameter for store stålkuler i olje, 10 til 38 mm [m]
d_iso = np.array([25, 39, 45, 55, 95, 115]) / 1000  # diameter for isoporkulene i luft, 25 mm til 115 mm [m]
term_staal = terminal_stokes(rho_kuler[2], rho_medium[1], mu_medium[1], d_array)
term_isopor = terminal_stokes(rho_kuler[0], rho_medium[0], mu_medium[0], d_array)

# analytiske verdier for terminalhastighet
teo_term_liten = terminal_stokes(rho_kuler[2], rho_medium[1], mu_medium[1], d_liten)
teo_term_stor = terminal_stokes(rho_kuler[2], rho_medium[1], mu_medium[1], d_stor)
teo_term_iso = terminal_stokes(rho_kuler[0], rho_medium[0], mu_medium[0], d_iso)
print("Analytiske hastigheter for 1. liten oljetank, 2. stor oljetank, 3. isopor i luft:")
print(teo_term_liten), print(teo_term_stor / 1000), print(teo_term_iso / 1000)  # mm/s, m/s, m/s

# teoretisk plot av hastigheter for stokes' strømning
plt.plot(d_array * 1e3, term_staal, '--', color='#9c9c9c', label='Stål i olje (S)')
plt.plot(d_array * 1e3, term_isopor, '--', color='#4f4f4f', label='Isopor i luft (S)')
plot_analyse()


# Data fra labben
stor_tank = np.array([197, 192, 190])
frames_67 = np.array([15, 9, 7, 7, 6, 5])  # bilder per sekund mellom 6 - 7 på stor tank
frames_78 = np.array([14, 9, 7, 6, 5, 4])  # bilder per sekund mellom 7 - 8 på stor tank
frames_89 = np.array([14, 9, 6, 6, 5, 4])  # bilder per sekund mellom 8 - 9 på stor tank
frames_300 = np.array([73, 20, 10, 5, 3, 2.5])  # bilder per sekund mellom 300 - 250 ml på liten tank
frames_250 = np.array([73, 19, 9, 5, 3, 2])  # bilder per sekund mellom 250 - 200 ml på liten tank
frames_200 = np.array([71, 19, 9.5, 5, 3, 2])  # bilder per sekund mellom 200 - 150 ml på liten tank
frames_iso = np.array([23, 21, 17, 16, 13, 6])  # bilder per sekund mellom toppen av stokken til bunnen av stokken

# Terminal hastigheter sett fra videoer
v_67, v_78, v_89 = terminal_video(frames_67, stor_tank[0]), terminal_video(frames_78, stor_tank[1]), \
                   terminal_video(frames_89, stor_tank[2])
print("Terminal hastigheter fra målinger på stor oljetank:")
print(v_78)  # Terminal hastigheter fra målinger vi bruker videre (Hvorfor akkurat disse?)

v_300, v_250, v_200 = terminal_video(frames_300, 76), terminal_video(frames_250, 76), terminal_video(frames_200, 76)
print("Terminal hastigheter fra målinger på liten oljetank:")
print(v_200)  # Terminal hastigheter fra målinger vi bruker videre (Hvorfor akkurat disse?)

v_iso = terminal_video(frames_iso, 201)  # lengden på stokken som ble brukt til å måle er 201 [mm]
print("Terminal hastigheter fra målinger på isopor i luft")
print(v_iso)  # Terminal hastigheter fra målinger vi bruker videre (Hvorfor akkurat disse?)

# Lineær regresjoner av dataene fra stål i olje og isopor i luft (begge akser gitt i mm)
stig_liten, err_liten, konst_liten = linear_regression(d_liten[0:2] * 1e3, v_200[0:2], "stål i liten olje")
stig_stor, err_stor, konst_stor = linear_regression(d_stor * 1e3, v_78, 'Regresjon av stål i stor olje')
stig_iso, err_iso, konst_iso = linear_regression(d_iso * 1e3, v_iso, "isopor i luft")

print(stokes_coefficient(mu_medium[1], rho_medium[1], rho_kuler[2], abs(konst_liten)))
print(stokes_coefficient(mu_medium[1], rho_medium[1], rho_kuler[2], konst_stor))
print(stokes_coefficient(mu_medium[1], rho_medium[1], rho_kuler[2], konst_iso))

y_liten = (stig_liten * (d_liten[0:2] * 1e3)) + konst_liten
y_stor = (stig_stor * (d_array * 1e3)) + np.log10(konst_stor)
y_iso = (stig_iso * (d_array * 1e3)) + np.log10(konst_iso)
y2 = (d_array * 1e3) ** 2
y15 = (d_array * 1e3) ** 0.5 * 1e2

plt.plot(d_liten[0:2] * 1e3, y_liten, label='Lin.reg. små kuler')
plt.plot(d_array * 1e3, y_stor, label='Lin.reg. store kuler')
plt.plot(d_array * 1e3, y_iso, label='Lin.reg. isopor')
plt.plot(d_array * 1e3, y2, '--', label='Stokes')
plt.plot(d_array * 1e3, y15, '--', label='Rayleigh')

# Hastighetsplott av alle målinger som ble gjort for alle kuler (for å velge ut beste målinger)
# plot_sammenligning(d_stor, v_67, 'S: 6 - 7'), plot_sammenligning(d_stor, v_78, 'S: 7 - 8')
# plot_sammenligning(d_stor, v_89, 'S: 8 - 9'), plot_sammenligning(d_liten, v_300, 'L: 300 - 250')
# plot_sammenligning(d_liten, v_250, 'L: 250 - 200'), plot_sammenligning(d_liten, v_200, 'L: 200 - 150')
plot_sammenligning(d_iso, v_iso, 'Isopor i luft'), plot_sammenligning(d_stor, v_78, 'S: 7 - 8')
plot_sammenligning(d_liten, v_200, 'L: 200 - 150')
# plt.savefig('innlev.png')
plt.show()


