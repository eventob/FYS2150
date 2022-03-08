import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def terminal_rayleigh(f_d, f_g, rho, r, vel):
    """
    (Turbulent fall)
    Funksjonen regner ut terminalhastigheten til objektet ved Rayleigh strømning.
    """
    c_r = f_d / (rho * r ** 2 * vel ** 2)
    return np.sqrt(f_g / (c_r * rho * r ** 2))


def terminal_stokes(f_d, mass, grav, mu, r, vel):
    """
    (Laminært fall)
    Funksjonen regner ut terminalhastigheten til objektet ved Stokes strømning.
    """
    c_s = f_d / (mu * r * vel)
    return c_s, (mass * grav) / (c_s * mu * r)


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

rho_staal = 7905 - 886      # forskjellen i massetetthet mellom kulene og mediumet (stål og olje)
mu_olje = 0.190

data = np.loadtxt("terminal_hastighet_rdata.dat")
navn = [0, "Rayleigh", "Annen", "Stoke"]

linear_regression(data[0], data[1], "Stokes'")
terminal_video(6.5, 30)
terminal_plot(data, navn)
