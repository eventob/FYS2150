import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

file = open('RC_data.csv', 'r').readlines()[1:]

N = len(file)
volt = np.zeros(N)
freq = volt.copy()
for i in range(N):
    volt[i] = float(file[i].split(',')[0])
    freq[i] = float(file[i].split(',')[1])

plt.xscale('log')
plt.yscale('log')
plt.plot(freq, volt)
plt.grid()


line = linregress(freq[11:], volt[11:])
stigningstall = line.slope
konstantledd = line.intercept

print(np.sqrt(0.25 * 10))
print(0.025 * 10.523 / 100)
print(0.025 * 50.752 / 100)
print(((0.05 / 100.15) + (0.002 / 10.253)) * 9.768)
print(20 / (2 ** 16) * 1e3)
print(stigningstall, konstantledd)
print(len(volt[11:]))

log_vuvi = -0.5 * np.log10(1 + ((freq[11:] / (10 ** konstantledd)) ** 2))
print(log_vuvi)
print(np.exp(konstantledd))
plt.plot(freq[11:], 10 ** log_vuvi)
plt.show()
