# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:21:39 2022

@author: vibishar
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
"""


#med 10 til 15 grader, 9.45 grader
array([11226.95641558, 11241.60664144, 11263.58198022, 11190.33085093,
       11207.4227811 , 11197.65596386, 11197.65596386, 11197.65596386,
       11197.65596386, 11197.65596386, 11197.65596386, 11197.65596386,
       11197.65596386, 11197.65596386, 11197.65596386, 11197.65596386,
       11197.65596386, 11197.65596386, 11197.65596386, 11197.65596386,
       11197.65596386, 11200.09766817, 13287.75485289, 11422.29276035])

array([0.20480051, 0.61440154, 1.02400256, 1.43360358, 1.84320461,
       2.25280563, 2.66240666, 3.07200768, 3.4816087 , 3.89120973,
       4.30081075, 4.71041178, 5.1200128 , 5.52961382, 5.93921485,
       6.34881587, 6.7584169 , 7.16801792, 7.57761894, 7.98721997,
       8.39682099, 8.80642202, 9.21602304, 9.62562406])



#take 3 vinkel 6.5
array([0.20480171, 0.61440512, 1.02400853, 1.43361195, 1.84321536,
       2.25281877, 2.66242219])

array([11204.98107679, 11212.30618972, 11219.63130265, 11226.95641558,
       11234.28152851, 11239.16493713, 11190.33085093])

time = np.array([0.20480171, 0.61440512, 1.02400853])
set1 = np.array([11226.95641558, 11241.60664144, 11263.58198022])
set2 = np.array([11224.51471127, 11244.04834575, 11263.58198022])

#vinkel 3.5
array([0.20480102, 0.61440307, 1.02400512, 1.43360717, 1.84320922,
       2.25281126, 2.66241331, 3.07201536, 3.48161741, 3.89121946,
       4.3008215 , 4.71042355])

array([11190.33085093, 11214.74789403, 11219.63130265, 11226.95641558,
       11236.72323282, 11190.33085093, 11190.33085093, 11197.65596386,
       11204.98107679, 11197.65596386, 11200.09766817, 11197.65596386])
"""


def vel(f, temp):
    f_m = 11198
    c = 331.1 + (0.606 * temp)
    return c * (1 - (1/(f_m / f)))


time = np.array([0.20480171, 0.61440512, 1.02400853])
hastighet_102 = np.array([-0.69, -1.38, -2.08])
hastighet_69 = vel(np.array([11204, 11212, 11219]), 21.8)
hastighet_56 = vel(np.array([11199, 11214, 11219]), 21.8)

reg_1 = stats.linregress(time, hastighet_102)
reg_2 = stats.linregress(time, hastighet_69)
reg_3 = stats.linregress(time, hastighet_56)

print(reg_1.slope, reg_2.slope, reg_3.slope)
print(reg_1.stderr, reg_2.stderr, reg_3.stderr)

x = np.linspace(0, 1.02400853, 50)      # same registered time for all angles
y_1 = reg_1.slope*x + reg_1.intercept
y_2 = reg_2.slope*x + reg_2.intercept
y_3 = reg_3.slope*x + reg_3.intercept

a_t = np.array([1.5862, 0.5608, 0.3036])
a_v = np.array([1.697, 0.56, 0.75])
a_theo = np.array([1.737, 1.179, 0.953])

plt.errorbar(a_t, a_theo, xerr=[0.04, 0.009, 0.004], label="Measurements")
plt.errorbar(a_v, a_theo, xerr=[0.07, 0.02, 0.22], label="Linear regression")
plt.errorbar(a_theo, a_theo, xerr=[0.019, 0.028, 0.035], label="Theoretic")
plt.xlabel("Measured acc [$m/s^2$]")
plt.ylabel("Theoretic acc [$m/s^2$]")
plt.title("Målt mot teoretisk akselerasjon")
plt.grid()
plt.legend()
plt.savefig("akselerasjon_plot.png")
plt.show()

