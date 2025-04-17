from tkinter import filedialog
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# def load_and_align_audio(path1, path2, sr=22050):
def load_and_align_audio(path1, path2, sr=40000):
    y1, _ = librosa.load(path1, sr=sr)
    y2, _ = librosa.load(path2, sr=sr)

    # Normalize
    y1 = y1 / np.max(np.abs(y1))
    y2 = y2 / np.max(np.abs(y2))

    # Align using cross-correlation
    corr = correlate(y2, y1, mode='full')
    lag = np.argmax(corr) - len(y1) + 1
    if lag > 0:
        y2 = y2[lag:]
    else:
        y1 = y1[-lag:]

    # Trim to same length
    min_len = min(len(y1), len(y2))
    return y1[:min_len], y2[:min_len], sr

def compute_spectrogram_from_signals(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    return S, freqs, times

def plot_spectrograms_top_bottom(spec1, spec2, freqs, times, file_name):
    plt.figure(figsize=(10, 8))

    # Original audio on top
    plt.subplot(2, 1, 1)
    plt.imshow(librosa.amplitude_to_db(spec1, ref=np.max), aspect='auto', origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.title(f"Original - {file_name}")
    plt.ylabel("Frequency (Hz)")

    plt.yscale('log')
    plt.ylim(100, 20000)
    custom_ticks = [100, 200, 400, 800, 1600, 3200, 6400, 12800,20000]
    plt.yticks(custom_ticks, labels=[str(t) for t in custom_ticks])
    
    plt.colorbar(format='%+2.0f dB')

    # Recorded audio on bottom
    plt.subplot(2, 1, 2)
    plt.imshow(librosa.amplitude_to_db(spec2, ref=np.max), aspect='auto', origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.title(f"Recorded - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.yscale('log')
    plt.ylim(100, 20000)
    custom_ticks = [100, 200, 400, 800, 1600, 3200, 6400, 12800,20000]
    plt.yticks(custom_ticks, labels=[str(t) for t in custom_ticks])

    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

def compute_accuracy(spec1, spec2):
    min_time_bins = min(spec1.shape[1], spec2.shape[1])
    spec1 = spec1[:, :min_time_bins]
    spec2 = spec2[:, :min_time_bins]

    max_val = np.maximum(spec1, spec2) + 1e-10
    error = np.abs(spec1 - spec2)
    accuracy = 1 - (error / max_val)
    accuracy_per_freq = np.mean(accuracy, axis=1) * 100
    return accuracy_per_freq

def process_all(original_folder, recorded_folder):
    filenames = sorted(os.listdir(original_folder))
    all_accuracies = []

    for file in filenames:
        if file.endswith(".mp3") and file in os.listdir(recorded_folder):
            orig_path = os.path.join(original_folder, file)
            rec_path = os.path.join(recorded_folder, file)

            y_orig, y_rec, sr = load_and_align_audio(orig_path, rec_path)

            spec_orig, freqs, times = compute_spectrogram_from_signals(y_orig, sr)
            spec_rec, _, _ = compute_spectrogram_from_signals(y_rec, sr)

            # Plot aligned spectrograms top-by-bot
            plot_spectrograms_top_bottom(spec_orig, spec_rec, freqs, times, file)

            # Plot accuracy curve
            accuracy = compute_accuracy(spec_orig, spec_rec)
            all_accuracies.append(accuracy)

            plt.figure(figsize=(10, 4))
            plt.plot(freqs, accuracy)
            plt.title(f"Accuracy by Frequency - {file}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Accuracy (%)")

            plt.xscale('log')
            plt.xlim(100, 20000)
            custom_ticks = [100, 200, 400, 800, 1600, 3200, 6400, 12800,20000]
            plt.xticks(custom_ticks, labels=[str(t) for t in custom_ticks])

            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            plt.show()

    if all_accuracies:
        avg_accuracy = np.mean(all_accuracies, axis=0)
        print(f'Average Accuracy is: {np.mean(avg_accuracy)}')
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, avg_accuracy, color='purple')
        plt.title(f"Average Accuracy by Frequency (All Files)\nTotal Average Accuracy is: {np.mean(avg_accuracy)}%")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Accuracy (%)")

        plt.xscale('log')
        plt.xlim(100, 20000)
        custom_ticks = [100, 200, 400, 800, 1600, 3200, 6400, 12800,20000]
        plt.xticks(custom_ticks, labels=[str(t) for t in custom_ticks])

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plt.show()
    else:
        print("No matching MP3 files found.")

# Run the process
original_sound_folder = filedialog.askdirectory(title="Select Folder with Original Sounds")
recorded_sound_folder = filedialog.askdirectory(title="Select Folder with Recorded Sounds")
process_all(original_sound_folder, recorded_sound_folder)