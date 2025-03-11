import numpy as np
import matplotlib.pyplot as plt

def main():
    class Microphone_Laser:
        def __init__(self):
            self.light_speed = 3e8 # vitesse lumiere
            self.lambda_ = 500e-9 # Lambda laser en metres
            self.detector_sampling_rate = 5e6 # Sampling Rate du detecteur en Hz

        def simulate_raw_intensity(self, time_data, displacement_data):
            # Simulate Intensity from interference
            E_raw = np.sin(2 * np.pi * self.light_speed * time_data / self.lambda_) + np.sin(2 * np.pi * self.light_speed / self.lambda_ * time_data +  2 * np.pi / self.lambda_ * displacement_data)
            I_raw = E_raw**2
            return time_data, I_raw
        
        def simulate_detector_response(self, time_data, displacement_data):
            time_raw, I_raw = self.simulate_raw_intensity(time_data, displacement_data)
            detector_response_time = 1/self.detector_sampling_rate
            t_sampled = np.arange(0, time_raw[-1], 1 / self.detector_sampling_rate)
            I_sampled = np.interp(t_sampled, time_raw, I_raw)  # Interpolate high-resolution data
            I_avg = []
            for t in t_sampled:
                mask = (time_raw >= t - detector_response_time / 2) & (time_raw <= t + detector_response_time / 2)
                I_avg.append(np.mean(I_raw[mask]))
            I_avg = np.array(I_avg)
            return t_sampled, I_avg
        
        def extract_sound_from_detector_response(self, detector_response):
            I_avg_max = np.max(detector_response)
            I_avg_min = np.min(detector_response)
            A_param = (np.sqrt(I_avg_max)+np.sqrt(I_avg_min))/np.sqrt(2)
            B_param = (np.sqrt(I_avg_max)-np.sqrt(I_avg_min))/np.sqrt(2)
            signal = (2*detector_response-A_param**2-B_param**2)/(2*A_param*B_param)
            signal  = np.arccos(np.clip(signal, -1, 1))
            signal = self.convolve(signal, 5) # 5e-5 / t_sampled[1] donne la periode min du hearing range humain
            signal = np.diff(signal)
            signal = abs(signal)
            signal[::2] *= -1
            return signal
        
        def convolve(self, signal, size=5):
            kernel = np.ones(size) / size
            return np.convolve(signal, kernel, mode='same')
     
    def Theorical_Intensity_at_Detecteur(self, t, A, B, C, Freq_son, lambda_las):
        return ((A**2 + B**2)+2*A*B*np.cos(2*np.pi/lambda_las*C*np.sin(2*np.pi*Freq_son*t)))/2
    # theorical_detecteur_reponse = Theorical_Intensity_at_Detecteur(t_sampled, 1, 1, A_son, F_son, lambda_)

    def Membrane_displacement(t, A_son, F_son):
        return  A_son* np.sin(2 * np.pi * F_son * t)

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

    # My_Simu = Microphone_Laser()
    # F_son = 3000 # Freq du son en Hz
    # duration = 5/F_son # Dure de la simulation (s)
    # n =4.13 # facteur amplitude des oscillation de la mambrane vs lamda laser
    # A_son = n*My_Simu.lambda_ # Amplitude max des oscillation de la membrane en metre
    # time_raw = np.linspace(0, duration, int(np.pi/np.e*100000)) # Nombre random pour pas avoir de phenomene fucke de pas multiple de frequence de fonctions
    # Sound_data = Membrane_displacement(time_raw, A_son, F_son)

    # _, I_raw = My_Simu.simulate_raw_intensity(time_raw, Sound_data)
    # t_sampled, I_avg =  My_Simu.simulate_detector_response(time_raw, Sound_data)
    # signal = My_Simu.extract_sound_from_detector_response(I_avg)

    # plt.figure(figsize=(10, 5))
    # plt.plot(time_raw, I_raw, label='Intensité théorique', alpha=0.5)
    # plt.scatter(t_sampled, I_avg, color='blue', label='Intensité moyenne (réponse du détecteur)', marker='x')
    # # plt.plot(t_sampled, Theorical_Intensity_at_Detecteur(t_sampled, 1, 1, A_son, F_son, ), label='Enveloppe')
    # plt.plot(t_sampled[0:len(signal)], signal, label='signal Extrait de la mesure')
    # plt.xlabel('Temps (s)')
    # plt.ylabel('Intensité (a.u.)')
    # plt.title("Approximation de la mesure d'interférométrie au detecteur")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
