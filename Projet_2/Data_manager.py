# from Simulate_Laser_Mircophone import Microphone_Laser
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tkinter import filedialog
import sounddevice as sd
import soundfile as sf
import pandas as pd
import io
import glob
import os
import re
import scipy.signal as sgnl
import time
import scipy.io.wavfile as wav
from scipy.ndimage import maximum_filter1d, minimum_filter1d

def choose_file(file_type=None):
    if file_type == 'wav':
        filetypes = [("WAV Files", "*.wav")]
    if file_type == 'csv':
        filetypes=[("CSV Files", "*.csv")]
    if file_type == 'mp3':
        filetypes=[("MP3 Files", "*.mp3")]

    file_path = filedialog.askopenfilename(title="Select a WAV File", filetypes=filetypes)
    return file_path

def choose_wav_file():
    file_path = filedialog.askopenfilename(title="Select a WAV File", filetypes=[("WAV Files", "*.wav")])
    return file_path

def extract_time_data_from_wav(wav_file):
    if not wav_file:
        print("No file selected. Exiting.")
        return
    sample_rate, samples = wavfile.read(wav_file)
    # If stereo, convert to mono
    if samples.ndim > 1:
        samples = samples[:, 0]  # Keep only the left channel
    time = np.linspace(0, len(samples) / sample_rate, num=len(samples))  # Time axis
    return time, samples

def plot_waveform(time_raw, samples):
    plt.figure(figsize=(10, 5))
    plt.plot(time_raw, samples/np.max(samples))
    plt.xlabel('Temps (s)')
    plt.ylabel('Signal')
    plt.grid(True)
    plt.show()

def plot_data(signal):
    plt.figure(figsize=(10, 5))
    plt.plot(signal)
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    plt.grid(True)
    plt.title("Signal Plot")
    plt.show()

# def local_maximum(signal, range_for_max):
#     local_max = []
#     for i in range(len(signal)):
#         local_max.append(np.max(signal[max(i-range_for_max,0):min(i+range_for_max,len(signal))]))
#     return np.array(local_max)
# def local_minimum(signal, range_for_min):
#     local_min = []
#     for i in range(len(signal)):
#         local_min.append(np.min(signal[max(i-range_for_min,0):min(i+range_for_min,len(signal))]))
#     return np.array(local_min)


def local_maximum(signal, range_for_max):
    return maximum_filter1d(signal, size=2*range_for_max+1, mode='nearest')

def local_minimum(signal, range_for_min):
    return minimum_filter1d(signal, size=2*range_for_min+1, mode='nearest')

def extract_sound_from_detector_response(detector_response):
    detector_response = np.array(detector_response)
    # return detector_response-np.mean(detector_response)

    I_avg_max = np.max(detector_response)
    I_avg_min = np.min(detector_response)
    A_param = (np.sqrt(I_avg_max)+np.sqrt(I_avg_min))/np.sqrt(2)
    B_param = (np.sqrt(I_avg_max)-np.sqrt(I_avg_min))/np.sqrt(2)
    
    # I_avg_max = local_maximum(detector_response, 100000)
    # I_avg_min = local_minimum(detector_response, 100000)
    # A_param = (np.sqrt(I_avg_max)+np.sqrt(I_avg_min))/np.sqrt(2)
    # B_param = (np.sqrt(I_avg_max)-np.sqrt(I_avg_min))/np.sqrt(2)

    signal = detector_response
    plot_data(signal)
    signal = (2*signal-A_param**2-B_param**2)/(2*A_param*B_param)
    plot_data(signal)
    signal = np.clip(signal, -1, 1)
    plot_data(signal)
    signal = np.arccos(signal)
    plot_data(signal)
    signal = np.diff(signal)
    plot_data(signal)

    signal = abs(signal) #Degrade la qualite du son
    return signal

def convolve(signal, size=5):
    kernel = np.ones(size) / size
    return np.convolve(signal, kernel, mode='same')

def chose_data():
    # file_path = filedialog.askopenfilename(title="Select a HighRes.csv or Traces.csv File",filetypes=[("CSV Files", "*.csv")])
    file_path = choose_file(file_type='csv')
    return load_data_from_csv(file_path)

def load_data_from_csv(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the first non-metadata line (the header)
    for i, line in enumerate(lines):
        if not line.startswith("%"):
            header = lines[i-1].strip().split(", ")
            data_lines = lines[i + 1 :]
            break

    # Read the data into a Pandas DataFrame using io.StringIO
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), names=header)

    # Extract time and data channels
    time = df[header[0]].values  # Time column
    if len(header)>1:
        data = df[header[1]].values  # Data column
    else:
        data = time
        time = None

    return time, data

def select_view_and_play_wav():
    # wav_file = choose_wav_file()
    wav_file = choose_file(file_type='wav')
    time_raw, samples = extract_time_data_from_wav(wav_file)
    sd.play(samples, samplerate=11025)
    sd.wait()
    plot_waveform(time_raw, samples)


def select_and_play_csv():
    # wav_file = choose_wav_file()
    csv_file = choose_file(file_type='csv')
    time_raw, data_raw = load_data_from_csv(csv_file)
    sd.play(data_raw, samplerate=11025)
    sd.wait()

def choose_folder():
    folder_path = filedialog.askdirectory(title="Select a Folder Containing CSV Files")
    return folder_path

def plot_fourier_transform(time_raw, signal):
    # Compute the sampling rate
    dt = np.mean(np.diff(time_raw))  # Time step
    fs = 1 / dt  # Sampling frequency
    
    # Compute the Fourier Transform
    freq = np.fft.fftfreq(len(signal), d=dt)
    fft_values = np.fft.fft(signal)
    
    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(freq[:len(freq)//2], np.abs(fft_values[:len(freq)//2]))  # Plot only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Fourier Transform of the Signal")
    plt.grid()
    plt.show()

def filter_fourier_transform(time_raw, signal, a, b):
    # Compute the sampling rate
    dt = np.mean(np.diff(time_raw))
    
    # Compute the Fourier Transform
    freq = np.fft.fftfreq(len(signal), d=dt)
    fft_values = np.fft.fft(signal)
    
    # Apply frequency filtering
    filtered_fft = np.where((freq >= a) & (freq <= b), fft_values, 0)
    
    # Inverse Fourier Transform to get back the filtered signal
    filtered_signal = np.fft.ifft(filtered_fft)
    
    return np.real(filtered_signal)  # Return only real values

def save_as_wav(time_raw, signal):
    filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    
    if not filename:
        print("Saving canceled.")
        return
    
    # Compute the sampling rate
    dt = np.mean(np.diff(time_raw))
    fs = int(1 / dt)  # Convert to integer
    
    # Normalize the signal
    signal_normalized = np.int16(signal / np.max(np.abs(signal)) * 32767)
    
    # Save as WAV file
    wav.write(filename, fs, signal_normalized)
    print(f"File saved as {filename}")

def loop_trough_data_view_and_hear():
    # folder_path = choose_folder()
    folder_path = 'C:\\Users\\louis\\Documents\\ULaval_S4\\TPOP\\GitWorkSpace\\Projet_2\\Session3\\Test6'
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    pattern = r"(\d+Hz)(\d+Vpp).*_(\w+)\.csv"
    for file_path in csv_files:
        match = re.search(pattern, file_path.split("\\")[-1])
        if match:
            frequency = match.group(1)  # '100Hz'
            voltage = match.group(2)    # '10Vpp'
            file_type = match.group(3)  # 'HighRes'
            title = f"{frequency} {voltage} {file_type}"
        else:
            title = file_path.split("\\")[-1]
        # if file_type == 'HighRes':
        if 1:
        # if '10Vpp' in title:
            time_raw, mesure_raw = load_data_from_csv(file_path)
            # mesure_raw = convolve(mesure_raw, size=10)
            signal = extract_sound_from_detector_response(mesure_raw)
            # signal = signal[2:-2]

            # plt.figure(figsize=(10, 5))

            # plt.plot(time_raw[0:len(signal)], signal/np.max(signal), label = 'Treated signal')
            # plt.plot(time_raw, mesure_raw/np.max(mesure_raw), label = 'Mesure Raw', alpha=0.3)
            # plt.title(title)
            # plt.xlabel('Temps (s)')
            # plt.ylabel('Signal')
            # plt.grid(True)
            # plt.legend()
            # plt.show()

            # plot_fourier_transform(time_raw, signal)
            plot_data(signal)
            print(f'Filtering...')
            # signal = convolve(signal, size=30)
            signal = filter_fourier_transform(time_raw, signal, 75, 10000)
            print(f'Done!')
            # save_as_wav(time_raw, signal)
            plot_data(signal)


            # REsample if too high
            max_samplerate = 192000
            samplerate = int(len(signal) / (time_raw[-1] - time_raw[0]))
            # print(samplerate)
            if samplerate > max_samplerate:
                num_samples = int(len(signal) * max_samplerate / samplerate)
                signal = sgnl.resample(signal, num_samples)
                samplerate = max_samplerate

            # boost_factor = 5
            # signal = np.clip(signal, -1.0, 1.0)
            # signal = signal * boost_factor

            print(f'Playing {title}')
            sd.play(signal, samplerate=samplerate)
            sd.wait()
            print('Done!')


            # # Loop for 1 second to hear better
            # sd.play(signal, samplerate=samplerate, loop=True)
            # time.sleep(1)
            # sd.stop()


loop_trough_data_view_and_hear()
# select_view_and_play_wav()
# select_and_play_csv()
 



