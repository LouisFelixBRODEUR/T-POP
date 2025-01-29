import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os


# path = os.path.dirname(os.path.abspath(__file__))+"\\Mesure_1\\Lab_Bruit_Mesure_20250129_093708_HighRes.csv"
df = pd.read_csv(path)

temps = df["time"].to_numpy()
# chan_A = df["chan A"].to_numpy()
tension = df["chan B"].to_numpy()

# ------------------------------------------------------------
# 3. VISUALISER LE SIGNAL BRUT
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(temps, tension, color="red", label="Signal brut")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Signal mesuré par l'oscilloscope")
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# 4. FILTRAGE PAR MOYENNE GLISSANTE
# ------------------------------------------------------------
def moving_average(signal, window_size=10):
    """Retourne le signal filtré par moyenne glissante sur window_size points."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Ex.: on choisit une fenêtre de 20 points
window_size = 20
tension_mvg = moving_average(tension, window_size=window_size)

# Pour la partie temps, on ajuste la même longueur que le signal filtré
temps_mvg = temps[:len(tension_mvg)]

plt.figure(figsize=(10, 4))
plt.plot(temps, tension, alpha=0.4, label="Signal brut", color="red")
plt.plot(temps_mvg, tension_mvg, label="Moyenne glissante", color="blue")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Filtrage par moyenne glissante (window_size = {})".format(window_size))
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# 5. FILTRAGE PASSE-BAS BUTTERWORTH
# ------------------------------------------------------------
def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Applique un filtre passe-bas Butterworth d'ordre 'order' 
    avec fréquence de coupure 'cutoff_freq' sur le signal 'data'.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Estimation de la fréquence d'échantillonnage
# (hypothèse : points régulièrement espacés)
dt = np.diff(temps)  # intervalles de temps successifs
sample_rate = 1 / np.mean(dt)  # 1 / période moyenne

cutoff_freq = 50  # Fréquence de coupure (en Hz) à adapter
order = 4         # Ordre du filtre

tension_butter = butter_lowpass_filter(tension, cutoff_freq, sample_rate, order)

plt.figure(figsize=(10, 4))
plt.plot(temps, tension, alpha=0.4, label="Signal brut", color="red")
plt.plot(temps, tension_butter, label="Passe-bas Butterworth", color="blue")
plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Filtrage passe-bas Butterworth (fc = {} Hz, ordre = {})".format(cutoff_freq, order))
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# 6. DÉTECTION DES TRANSITIONS (par ex. 0V <-> 5V)
# ------------------------------------------------------------
# On suppose que le signal varie entre 0V et 5V. 
# Le seuil = 2.5V pour détecter les passages.

seuil = 2.5

# Détection des transitions montantes :
transitions_montantes = np.where((tension_butter[:-1] < seuil) & (tension_butter[1:] >= seuil))[0]

# Détection des transitions descendantes :
transitions_descendantes = np.where((tension_butter[:-1] >= seuil) & (tension_butter[1:] < seuil))[0]

# Plot des transitions détectées
plt.figure(figsize=(10, 4))
plt.plot(temps, tension_butter, label="Signal filtré (Butterworth)", color="blue")

# Marquer en noir les montées, en magenta les descentes
plt.scatter(temps[transitions_montantes], tension_butter[transitions_montantes],
            color="black", label="Transitions montantes", zorder=3)

plt.scatter(temps[transitions_descendantes], tension_butter[transitions_descendantes],
            color="magenta", label="Transitions descendantes", zorder=3)

plt.xlabel("Temps (s)")
plt.ylabel("Tension (V)")
plt.title("Détection des transitions 0V <-> 5V")
plt.legend()
plt.grid()
plt.show()

# Exemple : afficher les instants de transition dans la console
print("\nInstants des transitions montantes :")
for idx in transitions_montantes:
    print(f"- t = {temps[idx]:.6f}s")

print("\nInstants des transitions descendantes :")
for idx in transitions_descendantes:
    print(f"- t = {temps[idx]:.6f}s")

# ------------------------------------------------------------
# 7. SAUVEGARDE DES DONNÉES FILTRÉES
# ------------------------------------------------------------
# Exemple : on sauvegarde la version filtrée Butterworth

df_filtre = pd.DataFrame({
    "Temps (s)": temps,
    "Tension filtrée (V)": tension_butter
})

# df_filtre.to_csv("data_filtree_butterworth.csv", index=False)
# print("\nDonnées filtrées sauvegardées dans 'data_filtree_butterworth.csv'")
