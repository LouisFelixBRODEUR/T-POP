import numpy as np
import matplotlib.pyplot as plt

c = 3e8        # vitesse lumiere
lambda_ = 500e-9 # Lambda laser en metres
n=0.2 # facteur amplitude des oscillation de la mambrane vs lamda laser
A = n*lambda_   # Amplitude max des oscillation de la membrane en metre
F = 3000        # Freq du son en Hz

t = np.linspace(0, 0.005, int(1289*np.pi**np.e)) # Nombre random pour pas avoir de phenomene fucke de pas multiple de frequence de fonctions

y = (np.sin(2 * np.pi * c * t / lambda_) + np.sin(2 * np.pi * c * t / lambda_ + A*2*np.pi/lambda_ * np.sin(2 * np.pi * F * t)))**2

plt.plot(t, y)
plt.xlabel('Temps(s)')
plt.ylabel('Intensité')
plt.title('Approximation de la mesure au detecteur')
plt.grid(True)
plt.show()