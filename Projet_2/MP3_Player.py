from Simulate_Laser_Mircophone import Microphone_Laser
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tkinter import filedialog
import time
import sounddevice as sd
import soundfile as sf
import sounddevice as sd

def play_sound_old(time, samples):
    sample_rate = 1 / (time[1] - time[0])
    sd.play(samples, 1000)
    # sd.play(samples)
    sd.wait()

def play_sound(time, samples, target_rate=44100):
    # Compute original sample rate
    original_rate = 1 / (time[1] - time[0])

    # If original_rate is too high, resample the data
    if original_rate > target_rate:
        factor = int(original_rate // target_rate)
        samples = samples[::factor]  # Downsample
        time = time[::factor]  # Keep time aligned (optional)
    
    # Play with the adjusted sample rate
    sd.play(samples, samplerate=target_rate)
    sd.wait()

def choose_wav_file():
    file_path = filedialog.askopenfilename(title="Select a WAV File", filetypes=[("WAV Files", "*.wav")])
    return file_path

def plot_wav_waveform(wav_file):
    if not wav_file:
        print("No file selected. Exiting.")
        return
    sample_rate, samples = wavfile.read(wav_file)
    # If stereo, convert to mono
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    time = np.linspace(0, len(samples) / sample_rate, num=len(samples))  # Time axis
    return time, samples

My_simu = Microphone_Laser(sampling_rate_detector=5e6) # Base sampling_rate_detector = 5e6

# # Get sound from .wav
# wav_file = choose_wav_file()
# time_raw, samples = plot_wav_waveform(wav_file)
# # crop = 0.005 # sec
# # samples = samples[int(len(time_raw)/2-crop/2/time_raw[1]):int(len(time_raw)/2+crop/2/time_raw[1])] 
# # time_raw = time_raw[int(len(time_raw)/2-crop/2/time_raw[1]):int(len(time_raw)/2+crop/2/time_raw[1])]
# Sound_data = samples/np.max(abs(samples))*18.4*My_simu.lambda_

# Get sound from wave function
F_son = 300 # Freq du son en Hz
duration = 5/F_son # Dure de la simulation (s)
# duration = 1 # Dure de la simulation (s)
n = 300 # facteur amplitude des oscillation de la mambrane vs lamda laser (n=0.5@35lambda pour 3000Hz)(n=15@300lambda pour 300Hz)
A_son = n*My_simu.lambda_ # Amplitude max des oscillation de la membrane en metre
time_raw = np.linspace(0, duration, int(np.pi/np.e*duration*7.0531e7)) # Nombre random pour pas avoir de phenomene fucke de pas multiple de frequence de fonctions
Sound_data = A_son* np.sin(2 * np.pi * F_son * time_raw)

# Simulate sound a travers le setup
start = time.time()
_, I_raw = My_simu.simulate_raw_intensity(time_raw, Sound_data)
t_sampled, I_avg =  My_simu.simulate_detector_response(time_raw, Sound_data)
print(f'Detector response simulatated in {time.time()-start} sec')
start = time.time()
signal = My_simu.extract_sound_from_detector_response(I_avg)
print(f'Sound extracted from detector response in {time.time()-start} sec')

# Crop les bout
t_sampled = t_sampled[2:len(signal)-2]
signal = signal[2:-2]

play_sound(time_raw, Sound_data/np.max(Sound_data))
play_sound(t_sampled, signal/np.max(signal))



plt.figure(figsize=(10, 5))
plt.plot(time_raw, Sound_data/np.max(Sound_data), label="Raw Audio Signal (normalized)")
plt.plot(t_sampled, signal/np.max(signal), label='Sound extracted from simulation (normalized)')
# plt.plot(time_raw, membrane_displacement/np.max(membrane_displacement), label='membrane Displacement (normalized)')
plt.xlabel('Temps (s)')
plt.ylabel('Intensité (a.u.)')
plt.grid(True)
plt.legend()
plt.show()
