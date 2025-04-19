from tkinter import filedialog
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import convolve

def load_and_align_audio(path1, path2, path3=None, sr=40000):
    y1, _ = librosa.load(path1, sr=sr)
    y2, _ = librosa.load(path2, sr=sr)
    y3, _ = librosa.load(path3, sr=sr) if path3 else (None, None)

    # Normalize all signals
    y1 = y1 / np.max(np.abs(y1))
    y2 = y2 / np.max(np.abs(y2))
    if y3 is not None:
        y3 = y3 / np.max(np.abs(y3))

    # Align using cross-correlation (align y2 and y3 to y1)
    def align_to_reference(reference, signal):
        corr = correlate(signal, reference, mode='full')
        lag = np.argmax(corr) - len(reference) + 1
        if lag > 0:
            return signal[lag:]
        else:
            return signal[-lag:]

    y2 = align_to_reference(y1, y2)
    if y3 is not None:
        y3 = align_to_reference(y1, y3)

    # Trim to same length
    min_len = min(len(y1), len(y2), len(y3) if y3 is not None else np.inf)
    return (y1[:min_len], y2[:min_len], y3[:min_len] if y3 is not None else None, sr)

def compute_spectrogram_from_signals(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    return S, freqs, times

def plot_spectrograms_three_way(spec1, spec2, spec3, freqs, times, file_name):
    fig = plt.figure(figsize=(10, 12))
    
    # Tight grid layout with minimal margins
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.02], height_ratios=[1, 1, 1],
                        left=0.065, right=0.935, bottom=0.06, top=0.99,
                        hspace=0.13, wspace=0.02)
    
    # Font size configuration
    tick_fontsize = 18
    axis_label_fontsize = 20
    cbar_label_fontsize = 20
    
    # Custom ticks setup
    custom_ticks = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000]
    custom_tick_labels = ['100', '200', '400', '800', '1600', '3200', '6400', '12800', '20000']
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0]) if spec3 is not None else None
    
    # Plot parameters
    vmin, vmax = -80, 0
    
    # Plotting function to reduce repetition
    def plot_spectrogram(ax, data, title):
        im = ax.imshow(librosa.amplitude_to_db(data, ref=np.max), aspect='auto', origin='lower',
                      extent=[times[0], times[-1], freqs[0], freqs[-1]], vmin=vmin, vmax=vmax)
        # ax.set_title(title, fontsize=title_fontsize, pad=8)
        ax.set_yscale('log')
        ax.set_ylim(100, 20000)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_tick_labels, fontsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.minorticks_off()
        return im
    
    # Plot each spectrogram
    im1 = plot_spectrogram(ax1, spec1, "Original")
    im2 = plot_spectrogram(ax2, spec2, "Blue Microphone")
    
    if spec3 is not None:
        im3 = plot_spectrogram(ax3, spec3, "Michelphone")
    else:
        ax3.axis('off')
    
    # Shared colorbar - placed very close to plots
    cbar_ax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im1, cax=cbar_ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    # Big axis labels placed very close to plots
    fig.text(0.001, 0.5, 'Fréquence (Hz)', va='center', rotation='vertical', 
            fontsize=axis_label_fontsize)
    fig.text(0.5, 0.01, 'Temps (s)', ha='center', 
            fontsize=axis_label_fontsize)
    
    plt.show()

def process_all(folder1, folder2, folder3=None):
    filenames = sorted(os.listdir(folder1))
    all_accuracies_1_2 = []
    all_accuracies_1_3 = []
    all_accuracies_2_3 = []

    for file in filenames:
        if (file.endswith(".mp3") and 
            file in os.listdir(folder2) and 
            (folder3 is None or file in os.listdir(folder3))):
            
            path1 = os.path.join(folder1, file)
            path2 = os.path.join(folder2, file)
            path3 = os.path.join(folder3, file) if folder3 else None

            y1, y2, y3, sr = load_and_align_audio(path1, path2, path3)

            spec1, freqs, times = compute_spectrogram_from_signals(y1, sr)
            spec2, _, _ = compute_spectrogram_from_signals(y2, sr)
            spec3, _, _ = compute_spectrogram_from_signals(y3, sr) if y3 is not None else (None, None, None)

            # Plot spectrograms
            plot_spectrograms_three_way(spec1, spec2, spec3, freqs, times, file)
    else:
        print("No matching MP3 files found.")

# Run the process
# folder1 = filedialog.askdirectory(title="Select Folder with Audio 1")
# folder2 = filedialog.askdirectory(title="Select Folder with Audio 2")
folder3 = filedialog.askdirectory(title="Select Folder with Audio 3")
folder1 = "C:\\Users\\louis\\Documents\\ULaval_S4\\TPOP\\GitWorkSpace\\Projet_2\\Session6\\Original_Audio"
folder2 = "C:\\Users\\louis\\Documents\\ULaval_S4\\TPOP\\GitWorkSpace\\Projet_2\\Session6\\Recordings"
# folder3 = "C:\\Users\\louis\\Documents\\ULaval_S4\\TPOP\\GitWorkSpace\\Projet_2\\Session6\\Michelphone_MP3"
process_all(folder1, folder2, folder3)