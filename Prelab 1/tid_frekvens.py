import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import probplot
from scipy.stats import linregress


def py_qqplot():
    x = np.linspace(0, 99, 100)
    y = 2*x + np.random.randn(len(x))    #lager noen datapunkter med tilfeldig støy
    probplot(y, plot=plt)
    plt.show()


def linreg_val():
    x = np.linspace(0, 99, 100)
    y = 2 * x + np.random.randn(len(x))  # lager noen datapunkter med tilfeldig støy
    line = linregress(x, y)  # se dokumentasjon for hva linregress returnerer

    stigningstall = line.slope
    konstantledd = line.intercept
    standard_error_slope = line.stderr
    standard_error_intercept = line.intercept_stderr
    return stigningstall, konstantledd, standard_error_slope, standard_error_intercept


def data_std(arr, mean):
    return np.sqrt((1 / (len(arr[1:]) - 1)) * np.sum((arr - mean) ** 2))


def std_mean(std, length):
    return std / np.sqrt(length)


measurements = np.loadtxt("data1.csv")

N = len(measurements[1:])
mean = np.mean(measurements[1:])
s = data_std(measurements[1:], mean)

k = 0
for i in range(1, N):
    # ALLTID HUSK at MEAN må værra med for å finne verdier innenfor STANDARDAVVIKET
    if (mean - 2 * s) < measurements[i] < (mean + 2 * s):
        k += 1

mean_error = std_mean(s, N)
print(s)
print(mean)
print(mean_error)
print(k)

plt.hist(measurements, bins=10)
plt.axvline(mean, color='red')
plt.show()




# kode fra prelab
def mean(data):
    """
    finner gjennomsnitt
    """
    f = 1 / len(data) * np.sum(data)
    return f


def std(data):
    """
    Finner standardavvik
    """
    f = mean(data)
    s = np.sqrt(1 / (len(data) - 1) * np.sum((data - f) ** 2))
    return s


def mean_std(data):
    """
    Finner standardavviket i gjennomsnittet
    """
    return std(data) / np.sqrt(len(data))


n = np.array([2, 20, 40, 60, 80, 100, 150, 250, 500, 1000])
a = 1
b = 2
x_mid = np.zeros(len(n))
s = np.zeros(len(n))
sm = np.zeros(len(n))

random.seed(1)
for i in range(len(n)):
    x = a + b * np.random.randn(n[i], 1)
    x_mid[i] = mean(x)
    s[i] = std(x)
    sm[i] = mean_std(x)

plt.style.use("seaborn")
plt.plot(n, x_mid, label="Gjennomsnitt")
plt.plot(n, s, label="Standardavvik")
plt.plot(n, sm, label="Standardavviket i gjennomsnittet")
plt.legend()
plt.show()
