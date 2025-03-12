import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tkinter import filedialog
import time
import sounddevice as sd
import soundfile as sf
import sounddevice as sd

class Microphone_Laser:
    def __init__(self, longuer_donde_laser = 500e-9, sampling_rate_detector = 5e6):
        self.light_speed = 3e8 # vitesse lumiere
        self.lambda_ = longuer_donde_laser # Lambda laser en metres
        self.detector_sampling_rate = sampling_rate_detector # Sampling Rate du detecteur en Hz

    def simulate_raw_intensity(self, time_data, displacement_data):
        # Simulate Intensity from interference
        E_raw = np.sin(2 * np.pi * self.light_speed * time_data / self.lambda_) + np.sin(2 * np.pi * self.light_speed / self.lambda_ * time_data +  2 * np.pi / self.lambda_ * displacement_data)
        I_raw = E_raw**2
        return time_data, I_raw
    
    # def simulate_detector_response(self, time_data, displacement_data):
    #     time_raw, I_raw = self.simulate_raw_intensity(time_data, displacement_data)
    #     detector_response_time = 1/self.detector_sampling_rate
    #     t_sampled = np.arange(time_raw[0], time_raw[-1], detector_response_time)
    #     I_sampled = np.interp(t_sampled, time_raw, I_raw)  # Interpolate high-resolution data
    #     I_avg = []
    #     for t in t_sampled:
    #         mask = (time_raw >= t - detector_response_time / 2) & (time_raw <= t + detector_response_time / 2)
    #         I_avg.append(np.mean(I_raw[mask]))
    #     I_avg = np.array(I_avg)
    #     return t_sampled, I_avg
    
    def simulate_detector_response(self, time_data, displacement_data):
        time_raw, I_raw = self.simulate_raw_intensity(time_data, displacement_data)
        detector_response_time = 1/self.detector_sampling_rate
        t_sampled = np.arange(time_raw[0], time_raw[-1], detector_response_time)

        if len(time_raw) < len(t_sampled) or 2>int(detector_response_time/(time_raw[1]-time_raw[0])):
            print('len(time_raw) < len(t_sampled)')
            # I_sampled = self.interpolate_extrapolate(len(t_sampled), time_raw)
            I_sampled = self.convolve(I_raw, 2)
            I_sampled = np.interp(t_sampled, time_raw, I_sampled)  # Interpolate high-resolution data

        else:
            I_sampled = self.convolve(I_raw, int(detector_response_time/(time_raw[1]-time_raw[0])))
            I_sampled = np.interp(t_sampled, time_raw, I_sampled)  # Interpolate high-resolution data


        return t_sampled, I_sampled


    def interpolate_extrapolate(self, n, data):
        x_original = np.linspace(0, len(data) - 1, len(data))
        x_new = np.linspace(0, len(data) - 1, n)
        interpolated_data = np.interp(x_new, x_original, data)
        return list(interpolated_data)

    def extract_sound_from_detector_response(self, detector_response):
        detector_response = np.array(detector_response)
        I_avg_max = np.max(detector_response)
        I_avg_min = np.min(detector_response)
        A_param = (np.sqrt(I_avg_max)+np.sqrt(I_avg_min))/np.sqrt(2)
        B_param = (np.sqrt(I_avg_max)-np.sqrt(I_avg_min))/np.sqrt(2)
        signal = (2*detector_response-A_param**2-B_param**2)/(2*A_param*B_param)
        signal  = np.arccos(np.clip(signal, -1, 1))
        signal = self.convolve(signal, 5) # 5e-5 / t_sampled[1] donne la periode min du hearing range humain
        signal = np.diff(signal)
        signal = abs(signal)
        # signal[::2] *= -1
        return signal
    
    def convolve(self, signal, size=5):
        kernel = np.ones(size) / size
        return np.convolve(signal, kernel, mode='same')
    
def Theorical_Intensity_at_Detecteur(self, t, A, B, C, Freq_son, lambda_las):
    return ((A**2 + B**2)+2*A*B*np.cos(2*np.pi/lambda_las*C*np.sin(2*np.pi*Freq_son*t)))/2

def FFT(time_data, data):
    # Compute the sampling frequency
    dt = np.mean(np.diff(time_data))  # Time step (assuming uniform spacing)
    fs = 1 / dt  # Sampling frequency

    # Compute the FFT
    N = len(time_data)  # Number of points
    freqs = np.fft.fftfreq(N, d=dt)  # Frequency values
    fft_data = np.fft.fft(data)  # FFT of Channel A

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

def main():
    My_Simu = Microphone_Laser()

    F_son = 3000 # Freq du son en Hz
    duration = 5000/F_son # Dure de la simulation (s)
    # duration = 1 # Dure de la simulation (s)
    n =4.13 # facteur amplitude des oscillation de la mambrane vs lamda laser
    A_son = n*My_Simu.lambda_ # Amplitude max des oscillation de la membrane en metre
    time_raw = np.linspace(0, duration, int(np.pi/np.e*duration*7.0531e7)) # Nombre random pour pas avoir de phenomene fucke de pas multiple de frequence de fonctions
    Sound_data = A_son* np.sin(2 * np.pi * F_son * time_raw)

    # wav_file = choose_wav_file()
    # time_raw, samples = plot_wav_waveform(wav_file)
    # crop = 0.05 # sec
    # samples = samples[int(len(time_raw)/2-crop/2/time_raw[1]):int(len(time_raw)/2+crop/2/time_raw[1])] 
    # time_raw = time_raw[int(len(time_raw)/2-crop/2/time_raw[1]):int(len(time_raw)/2+crop/2/time_raw[1])]
    # Sound_data = samples/np.max(abs(samples))*1*My_Simu.lambda_


    # Simulate sound a travers le setup
    start = time.time()
    _, I_raw = My_Simu.simulate_raw_intensity(time_raw, Sound_data)
    t_sampled, I_avg =  My_Simu.simulate_detector_response(time_raw, Sound_data)
    print(f'Detector response simulatated in {time.time()-start} sec')
    start = time.time()
    signal = My_Simu.extract_sound_from_detector_response(I_avg)
    print(f'Sound extracted from detector response in {time.time()-start} sec')

    # plt.figure(figsize=(10, 5))
    # # plt.plot(time_raw, I_raw, label='Intensité théorique', alpha=0.5)
    # # plt.scatter(t_sampled, I_avg, color='blue', label='Intensité moyenne (réponse du détecteur)', marker='x')
    # # plt.plot(t_sampled, Theorical_Intensity_at_Detecteur(t_sampled, 1, 1, A_son, F_son, ), label='Enveloppe')
    # plt.plot(t_sampled[0:len(signal)], signal/np.max(np.abs(signal)), label='signal Extrait de la mesure')
    # plt.plot(time_raw, Sound_data/np.max(np.abs(Sound_data)), label='Original signal')
    # plt.xlabel('Temps (s)')
    # plt.ylabel('Intensité (a.u.)')
    # plt.title("Approximation de la mesure d'interférométrie au detecteur")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
