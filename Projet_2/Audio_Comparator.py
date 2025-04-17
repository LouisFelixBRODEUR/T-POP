import numpy as np
import matplotlib.pyplot as plt
import librosa
import tkinter as tk
from tkinter import filedialog
import os
from scipy.signal import savgol_filter

def select_audio_file(prompt):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("Audio Files", "*.wav *.mp3")])
    return file_path

def load_audio(file_path, sr=44100):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def align_signals(audio1, audio2):
    # Compute cross-correlation
    correlation = np.correlate(audio1, audio2, mode='full')
    lag = np.argmax(correlation) - len(audio2) + 1

    # Align audio2 to audio1
    if lag > 0:
        audio2_aligned = np.pad(audio2, (lag, 0), mode='constant')[:len(audio1)]
    else:
        audio2_aligned = audio2[-lag:]
        if len(audio2_aligned) < len(audio1):
            audio2_aligned = np.pad(audio2_aligned, (0, len(audio1) - len(audio2_aligned)), mode='constant')
        else:
            audio2_aligned = audio2_aligned[:len(audio1)]
    return audio2_aligned

def compute_fft(audio, sr):
    n = len(audio)
    fft_result = np.fft.rfft(audio)
    magnitude = np.abs(fft_result)
    freq = np.fft.rfftfreq(n, d=1/sr)
    return freq, magnitude

def compute_similarity(mag1, mag2):
    # Normalize magnitudes
    mag1_norm = mag1 / np.linalg.norm(mag1)
    mag2_norm = mag2 / np.linalg.norm(mag2)
    # Compute cosine similarity
    similarity = np.dot(mag1_norm, mag2_norm)
    return similarity

def plot_accuracy(freq, mag1, mag2, file1_name, file2_name):
    # Avoid division by zero
    mag1_safe = np.where(mag1 == 0, 1e-10, mag1)
    accuracy = 1 - np.abs(mag1 - mag2) / mag1_safe
    accuracy = np.clip(accuracy, 0, 1)  # Ensure values are between 0 and 1

    # Apply Savitzky-Golay filter for smoothing
    window_length = 101 if len(accuracy) >= 101 else len(accuracy) - (len(accuracy) + 1) % 2
    polyorder = 3  # Polynomial order for the filter

    if window_length >= polyorder + 2:
        accuracy_smooth = savgol_filter(accuracy, window_length=window_length, polyorder=polyorder)
    else:
        accuracy_smooth = accuracy  # If data is too short, skip smoothing

    plt.figure(figsize=(12, 6))
    plt.plot(freq, accuracy_smooth, label='Smoothed Reproduction Accuracy', color='blue')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xlim(20, 20000)  # Limit x-axis to the audible frequency range
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Accuracy')
    plt.title(f'Frequency-wise Reproduction Accuracy\n{file1_name} vs {file2_name}')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Select the original audio file.")
    file1 = select_audio_file("Select the original audio file")
    if not file1:
        print("No file selected. Exiting.")
        return

    print("Select the recorded audio file.")
    file2 = select_audio_file("Select the recorded audio file")
    if not file2:
        print("No file selected. Exiting.")
        return
    
    sr = 44100  # Sampling rate
    audio1 = load_audio(file1, sr)
    audio2 = load_audio(file2, sr)

    # Align audio2 to audio1
    audio2_aligned = align_signals(audio1, audio2)

    # Ensure both audio signals are the same length
    min_len = min(len(audio1), len(audio2_aligned))
    audio1 = audio1[:min_len]
    audio2_aligned = audio2_aligned[:min_len]

    freq1, mag1 = compute_fft(audio1, sr)
    freq2, mag2 = compute_fft(audio2_aligned, sr)

    # Ensure frequency arrays are the same
    if not np.array_equal(freq1, freq2):
        print("Frequency arrays do not match. Exiting.")
        return

    similarity = compute_similarity(mag1, mag2)
    print(f"Similarity Score (Cosine Similarity): {similarity:.4f}")

    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)
    plot_accuracy(freq1, mag1, mag2, file1_name, file2_name)

if __name__ == "__main__":
    main()
