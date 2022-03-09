import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def terminal_rayleigh(d):
    """
    (Turbulent fall)
    Funksjonen regner ut terminalhastigheten til objektet ved Rayleigh strømning.
    """
    # c_r = f_d / (rho * r ** 2 * vel ** 2)
    return d ** (1/2)


def terminal_stokes(rho_k, rho_m, mu, d):
    """
    (Laminært fall)
    Funksjonen regner ut terminalhastigheten til objektet ved Stokes strømning.
    """
    g = 9.81
    gamma = 1/18
    C_s = 6 * np.pi     # stokes-koeffisienten av en kule
    nu_s = mu / rho_k   # dynamisk viskositet over tetthet til mediet

    g_hat = ((rho_k / rho_m) - 1) * gamma * g
    return C_s, (g_hat / nu_s) * (d ** 2)


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
    stigningstall = 10 ** line.slope
    konstantledd = 10 ** line.intercept
    print("Gjennomsnittshastighet for %s strømning: %g +- %g [m/s]" % (label, stigningstall, konstantledd))
    return stigningstall, konstantledd


def terminal_video(frames, fps=60, length=25.5):
    """
    Funksjon for å beregne terminalhastighet fra videoer.
    """
    time_per_frame = 1 / fps
    fall_time = time_per_frame * frames
    terminal = length / fall_time
    print("Terminalhastighet fra video: %.5g [mm/s]" % terminal)
    return terminal


rho_kuler = [21.1, 1042, 7905]      # tetthet for (isopor, plast, stål) [kg / m^3]
rho_medium = [1.225, 889, 998]      # tetthet mediumer (luft, olje, vann) [kg / m^3]
nu_medium = [15.2 * 1e-6, 214 * 1e-6, 1.0 * 1e-6]        # viskositet mediumer (luft, olje, vann) [10^-6 m^2 / s]
mu_medium = [1.862 * 1e-5, 0.190, 0.998 * 1e-3]     # dynamisk viskositet (luft, olje, vann) [kg / ms]

d_stal = 7905 - 886      # forskjellen i massetetthet mellom kulene og mediumet (stål og olje)
mu_olje = 0.190

data = np.loadtxt("terminal_hastighet_rdata.dat")
navn = [0, "Rayleigh", "Annen", "Stoke"]

diameter = np.linspace(0.005, 0.1, 10000)       # [m]
c_s, stal_vel_s = terminal_stokes(rho_kuler[2], rho_medium[1], mu_medium[1], diameter)
# veloc_r = terminal_rayleigh(diameter)

# plt.plot(diameter, stal_vel_s)
# plt.plot(diameter, veloc)
# plt.xscale('log'), plt.yscale('log')
# plt.xlabel('Diameter [mm]'), plt.ylabel('Terminal hastighet $v_T (d)$ [mm/s]')
# plt.ylim(8, 1e4)
# plt.grid(which='both')
# plt.show()

# linear_regression(data[0], data[1], "Stokes'")
# terminal_video(6.5, 30)       # (frames, fps)
terminal_plot(data, navn)
