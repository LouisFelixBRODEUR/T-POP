import numpy as np
from scipy.io.wavfile import write
from tkinter import filedialog

def generate_sine_wave(frequency, sample_rate, duration, amplitude=1.0, phase=0):
    """
    Generate a sine wave for a given frequency, sample rate, and duration.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def create_harmonic_signal(frequencies, sample_rate=11025, duration=5.0):
    """
    Mix sine waves of given frequencies to create a harmonic signal.
    The frequencies should be provided in a list.
    """
    if not (1 <= len(frequencies)):
        raise ValueError("The number of frequencies must be 1 or higher.")

    # Start with an empty signal
    signal = np.zeros(int(sample_rate * duration))

    # Add each sine wave to the signal
    for frequency in frequencies:
        signal += generate_sine_wave(frequency, sample_rate, duration)
    
    # Normalize the signal to fit within the range of int16 for wav files
    signal = np.int16((signal / np.max(np.abs(signal))) * 32767)

    return signal

def save_signal_as_wav(signal, filename, sample_rate=11025):
    """
    Save the generated signal as a .wav file.
    """
    write(filename, sample_rate, signal)

def ask_save_location():
    file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    return file_path

# Example usage
if __name__ == "__main__":
    # Define the frequencies for the harmonic signal (in Hz)
    # frequencies = [110, 220, 330, 440]
    # frequencies = [120, 240, 360, 480]
    # frequencies = [180, 360, 540, 720]
    frequencies = [440, 660, 880, 1320]

    
    # Generate the harmonic signal
    harmonic_signal = create_harmonic_signal(frequencies)
    
    # Ask user for the save location
    save_path = ask_save_location()
    
    if save_path:
        # Save the harmonic signal as a .wav file at the selected location
        save_signal_as_wav(harmonic_signal, save_path)
        print(f"Harmonic signal saved as '{save_path}'")
    else:
        print("No file was selected.")