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

def extract_sound_from_detector_response(detector_response):
    detector_response = np.array(detector_response)
    I_avg_max = np.max(detector_response)
    I_avg_min = np.min(detector_response)
    A_param = (np.sqrt(I_avg_max)+np.sqrt(I_avg_min))/np.sqrt(2)
    B_param = (np.sqrt(I_avg_max)-np.sqrt(I_avg_min))/np.sqrt(2)
    signal = (2*detector_response-A_param**2-B_param**2)/(2*A_param*B_param)
    signal  = np.arccos(np.clip(signal, -1, 1))
    signal = np.diff(signal)
    signal = abs(signal)
    signal = convolve(signal, 10)
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

def loop_trough_data_view_and_hear():
    folder_path = choose_folder()
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    pattern = r"(\d+Hz)(\d+Vpp).*_(\w+)\.csv"
    for file_path in csv_files:
        match = re.search(pattern, file_path.split("\\")[-1])
        frequency = match.group(1)  # '100Hz'
        voltage = match.group(2)    # '10Vpp'
        file_type = match.group(3)  # 'HighRes'
        title = f"{frequency} {voltage} {file_type}"
        if file_type == 'HighRes':
            # time_raw, mesure_raw = chose_data()
            time_raw, mesure_raw = load_data_from_csv(file_path)
            signal = extract_sound_from_detector_response(mesure_raw)
            signal = signal[2:-2]


            plt.figure(figsize=(10, 5))

            plt.plot(time_raw[0:len(signal)], signal/np.max(signal), label = 'Treated signal')
            plt.plot(time_raw, mesure_raw/np.max(mesure_raw), label = 'Mesure Raw')
            plt.title(title)
            plt.xlabel('Temps (s)')
            plt.ylabel('Signal')
            plt.grid(True)
            plt.legend()
            plt.show()

            # REsample if too high
            # max_samplerate = 44100
            max_samplerate = 192000
            samplerate = int(len(signal) / (time_raw[-1] - time_raw[0]))
            if samplerate > max_samplerate:
                num_samples = int(len(signal) * max_samplerate / samplerate)
                signal = sgnl.resample(signal, num_samples)
                samplerate = max_samplerate

            # sd.play(signal, samplerate=samplerate)
            # sd.wait()

            # Loop for 1 second to hear better
            sd.play(signal, samplerate=samplerate, loop=True)
            time.sleep(1)
            sd.stop()


loop_trough_data_view_and_hear()
# select_view_and_play_wav()
# select_and_play_csv()
 



