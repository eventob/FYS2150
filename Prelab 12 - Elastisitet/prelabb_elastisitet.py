import numpy as np
import matplotlib.pyplot as plt
#from prelabb_masse_og_kraft import regression, linear
import soundfile as sf
import os

#spørsmål 4 og 5
data = np.loadtxt("maalinger_h.dat")  #kg, mm

m = data[:,0] #masse
h = data[:,1] #utslag
def regression(x,y):
    """
    Funksjon for å lage ekstrapoleringspunkter fra lineærtilpasningen
    """
    n = len(x)
    D = np.sum(x**2)-1/n*np.sum(x)**2
    E = np.sum(x*y)-1/n*np.sum(x)*np.sum(y)
    F = np.sum(y**2)-1/n*np.sum(y)**2

    x_bar = 1/n*np.sum(x)
    y_bar = 1/n*np.sum(y)

    m = E/D
    c = y_bar - m*x_bar
    d = y - m*x -c

    dm = np.sqrt(1/(n-2) *(D*F - E**2)/D**2)
    dc = np.sqrt(1/(n-2)*(D/n+x_bar**2)*(D*F - E**2)/D**2)
    return m, dm, c, dc
a, da, c, dc = regression(m,h)


print("A = {:.4e} m/kg".format(a/1000))
print("dA = {:.4e} m/kg".format(da/1000))


#spørsmål 8
#importerer soundfile for å lese av wav filer.
#pip install soundfile


f_hat = 1.107e3 #Hz
filename = "messing_lyd.wav"
mydata, samplerate = sf.read(filename)

duration = int(len(mydata)/samplerate)
#må sørge for at amplituden ikke overskrider 1 total
mydata*0.5


t = np.linspace(0,duration,len(mydata))
def sinus(f,t):
    return 0.5*np.sin(2*np.pi*f*t)

sum_signal = mydata + sinus(f_hat,t)

sf.write("sound.wav", sum_signal, samplerate)

#importerer playsound for å spille av det summerte signalet.
#pip install playsound

"""
Hvis lydfila har en path som er veldig lang, så får man en feilmelding.
Lagre fila i desktop
"""
#from playsound import playsound
#path = "C:\\Users\\jacob\\OneDrive\\Desktop"

#playsound(path + "\\svevninger.wav")
