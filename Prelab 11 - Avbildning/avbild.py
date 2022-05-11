import numpy as np
import matplotlib.pyplot as plt


print((11.5 ** 2 + 0.1 ** 2) / (2 * 0.1))

R = 280 / 1e3
n = 1.520
print(abs((n - 1) * ((-1/R) - (1/R)) * 1e3))


def r(x, d):
    return (x ** 2 + d ** 2) / (2 * d)
