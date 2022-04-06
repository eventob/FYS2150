import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson 
count = [4, 9, 3, 4, 5, 5, 2, 4, 4, 0, 2, 5, 2, 7, 4, 3, 6, 3, 0, 5, 0, 
         7, 4, 2, 3, 4, 3, 6, 2, 5, 2, 2, 1, 3, 4, 0, 3, 9, 2, 9, 4, 3
         ,3, 4, 5, 7, 3, 4, 3, 1, 2, 5, 3, 3, 5, 5, 1, 7, 4, 3, 5, 2, 
         1, 5, 6, 2, 4, 2, 1, 5, 3, 6, 3, 2, 4, 2, 1, 3, 4, 0, 2, 1, 
         2, 4, 5, 8, 6, 2, 5, 2, 1, 6, 3, 3, 6, 5, 4, 4, 5, 1, 3, 7, 
         4]

counts, bins = np.histogram(count)

#plt.xticks()
mu = np.mean(count)
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
plt.plot(x, poisson.pmf(x, mu) * 1e2, 'r-', ms=8, label='poisson pmf')

plt.hist(bins[:-1], bins, weights=counts, align = "left")

plt.grid(), plt.xlabel("Tellinger [$n_r$]"), plt.ylabel("Antall tellinger")
plt.savefig("histogram.png")

plt.show()

"""
print(np.average(count), np.std(count))

def std_dev(arr):
    
    #Funksjon som finner standardavviket til gjennomsnittet
    
    N = len(arr)
    avg_k = (1/N) * np.sum(arr)
    dev_std = np.sqrt((1 / (N - 1)) * np.sum((arr - avg_k) ** 2))
    print("Standard avviket til gjennomsnittet: %g" % dev_std)

std_dev(count)

print(np.sqrt(np.mean(count)))





print("Values of the probabilities:")

print(190*np.exp(-np.log(2)*19/30.2))

count60 = [106, 115,101, 134]
print("mean og 60s count", np.average(count60))
std_dev(count60)
count60_bak = [48, 31, 40, 31]
print("mean og 60s count", np.average(count60_bak))
std_dev(count60_bak)



def gm_eff(n_r, n_b, A, r, d):
    
    #Funksjon som regner effektiviteten til en gm-måler.
    
    omega = np.pi * r ** 2 / d ** 2
    gm = ((n_r - n_b) / (A * (omega / (4 * np.pi)))) * 1e2      # [%]
    print('GM_eff: %g' % gm)
    
    
gm_eff(114/60, 37.5/60, 122850, 0.011, 0.162)

""" 
    
    
    