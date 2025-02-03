import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.optimize import curve_fit
from matplotlib.ticker import EngFormatter

def main():
    #  Lecture du fichier CSV
    # path = os.path.dirname(os.path.abspath(__file__))+"\\Mesure_1\\Lab_Bruit_Mesure_20250129_093708_Traces.csv"
    # path = os.path.dirname(os.path.abspath(__file__))+"\\Mesure_1\\Lab_Bruit_Mesure_20250129_093708_HighRes.csv"
    path = os.path.dirname(os.path.abspath(__file__))+"\\Mesure_2\\MokuOscilloscopeData_20250129_124208_HighRes.csv"
    df = pd.read_csv(path)

    time = df["time"].to_numpy()
    chan_A = df["chan A"].to_numpy()
    chan_B = df["chan B"].to_numpy()
    index_neg_1 = list(time).index(-1)
    index_pos_1 = list(time).index(1)

    time = time[index_neg_1:index_pos_1]
    chan_A = chan_A[index_neg_1:index_pos_1]
    chan_B = chan_B[index_neg_1:index_pos_1]

    # chan_A = chan_A/3.3
    # Chan_B_mean = np.mean(chan_B)
    # chan_B = chan_B - Chan_B_mean
    # chan_B = chan_B
    # filtered_chan_B = LP_Filter(time, chan_B, 100)
    # chan_B = LP_Filter(time, chan_B, 200)
    # filtered_chan_B = filtered_chan_B
    

    # FFT(time, chan_B)

    val_B_Hight = []
    val_B_low = []
    val_B_fil_Hight = []
    val_B_fil_low = []
    for i, val in enumerate(chan_A):
        if val > 0.5:
            val_B_Hight.append(chan_B[i])
            # val_B_fil_Hight.append(filtered_chan_B[i])
        else:
            val_B_low.append(chan_B[i])
            # val_B_fil_low.append(filtered_chan_B[i])

    Val_chan_B_Hight = np.mean(val_B_Hight)
    Val_chan_B_Hight_STD = np.std(val_B_Hight)
    Val_chan_B_Low = np.mean(val_B_low)
    Val_chan_B_Low_STD = np.std(val_B_low)


    # Val_chan_B_fil_Hight = np.mean(val_B_fil_Hight)
    # Val_chan_B_fil_Low = np.mean(val_B_fil_low)
    # print(Val_chan_B_Hight)
    # print(Val_chan_B_Low)


    print((Val_chan_B_Hight - Val_chan_B_Low)/Val_chan_B_Low*100)
    # print((Val_chan_B_Low - Val_chan_B_Hight)/Val_chan_B_Hight*100)

    # Création de la figure et des axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Tracer la donnée principale
    ax.plot(time, chan_B, label="Signal du détecteur (DET36A2)", color="blue", linewidth=1)

    # Tracer les valeurs moyennes (haut et bas)
    ax.axhline(y=Val_chan_B_Hight, color="#46eb73", linestyle="-", linewidth=2, label="Moyenne signal haut")
    ax.axhline(y=Val_chan_B_Low, color="#ff8f17", linestyle="-", linewidth=2, label="Moyenne signal bas")

    # Tracer les zones d'incertitude (écart-type) avec un remplissage
    ax.fill_between(time, Val_chan_B_Hight - Val_chan_B_Hight_STD, Val_chan_B_Hight + Val_chan_B_Hight_STD, 
                    color="#46eb73", alpha=0.2, label="Incertitude signal haut")
    ax.fill_between(time, Val_chan_B_Low - Val_chan_B_Low_STD, Val_chan_B_Low + Val_chan_B_Low_STD, 
                    color="#ff8f17", alpha=0.2, label="Incertitude signal bas")

    # Ajouter les labels et le titre
    ax.set_xlabel("Temps", fontsize=25)
    ax.set_ylabel("Signal de sortie", fontsize=25)
    # ax.set_title("Signal de sortie du détecteur DET36A2 exposé par une led oscillante à 5Hz en fonction du temps", fontsize=14)


    ax.yaxis.set_major_formatter(EngFormatter(unit='V')) 
    formatter = EngFormatter(unit="s", places=3)  # places=3 ensures "ms"
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticks(ax.get_xticks())  # Refresh ticks
    ax.set_xticklabels([f"{tick * 1e3:.0f} ms" for tick in ax.get_xticks()])  # Convert to ms

    ax.set_ylim(0.012, 0.0185)
    ax.set_xlim(-1.1, 1.1)

    # Add text
    a=0
    b=0.00073
    equation = f"Signal haut = {round(Val_chan_B_Hight*1000,2)}±{round(Val_chan_B_Hight_STD*1000,2)}mV"
    x_min, x_max = chan_B.min(), chan_B.max()
    y_min, y_max = time.min(), time.max()
    plt.text(
        -0.5+a,
        0.0173+b,
        equation,
        fontsize=25,
        color="#46eb73"
    )
    equation = f"Signal bas = {round(Val_chan_B_Low*1000,2)}±{round(Val_chan_B_Low_STD*1000,2)}mV"
    x_min, x_max = chan_B.min(), chan_B.max()
    y_min, y_max = time.min(), time.max()
    plt.text(
        -0.5+a,
        0.017+b,
        equation,
        fontsize=25,
        color="#ff8f17"
    )

    # Ajouter une grille
    ax.grid(True, linestyle='--', alpha=0.6)

    # Ajouter une légende
    ax.legend(fontsize=15, loc='upper left')

    # Ajuster la taille des étiquettes des axes
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Afficher le graphique
    # plt.tight_layout()
    fig.subplots_adjust(left=0.085, right=0.99, top=0.99, bottom=0.085)
    plt.show()

def LP_Filter(time_data, data, cutoff):
    # Compute sampling frequency
    dt = np.mean(np.diff(time_data))  # Time step
    fs = 1 / dt  # Sampling frequency
    N = len(time_data)  # Number of points

    # Compute FFT
    fft_A = np.fft.fft(data)
    freqs = np.fft.fftfreq(N, d=dt)  # Frequency axis

    # for i, val in enumerate(fft_A):
    #     if val < 0.00003:
    #         fft_A[i] = 0

    # Apply low-pass filter (remove frequencies > 70 Hz)
    fft_A[np.abs(freqs) > cutoff] = 0  # Zero out high frequencies

    # Compute inverse FFT to get the filtered signal
    filtered_A = np.fft.ifft(fft_A).real
    return filtered_A

def FFT(time_data, data):
    # Compute the sampling frequency
    dt = np.mean(np.diff(time_data))  # Time step (assuming uniform spacing)
    fs = 1 / dt  # Sampling frequency

    # Compute the FFT
    N = len(time_data)  # Number of points
    freqs = np.fft.fftfreq(N, d=dt)  # Frequency values
    fft_data = np.fft.fft(data)  # FFT of Channel A

    # for i, val in enumerate(fft_data):
    #     if val < 0.00003:
    #         fft_data[i] = 0


    # Keep only the positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    fft_data = np.abs(fft_data[mask]) / N  # Normalize magnitude

    # Plot the FFT
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, fft_data, label="FFT", color="b")

    # Labels and formatting
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Frequency Spectrum")
    ax.legend()
    ax.grid()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
