import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.optimize import curve_fit

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

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the data
    # ax.plot(time, chan_A, label="Channel A", color="b", linewidth=1)
    ax.plot(time, chan_B, label="Channel B", color="r", linewidth=0.5)
    # ax.plot(time, filtered_chan_B, label="Filtered Channel B", color="green", linewidth=1)

    ax.axhline(y=Val_chan_B_Hight, color="red", linestyle="--", linewidth=2, label="chan B High")
    ax.axhline(y=Val_chan_B_Low, color="red", linestyle="--", linewidth=2, label="chan B Low")

    # ax.axhline(y=Val_chan_B_fil_Hight, color="green", linestyle="--", linewidth=2, label="chan B High")
    # ax.axhline(y=Val_chan_B_fil_Low, color="green", linestyle="--", linewidth=2, label="chan B Low")


    # Add labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Output Read")
    ax.set_title("Channel A and B Signals Over Time")
    ax.legend()

    # Show the plot
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
