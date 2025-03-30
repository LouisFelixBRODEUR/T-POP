import numpy as np
import soundfile as sf
import os
import tkinter as tk
from tkinter import filedialog

def choose_audio_file():
    """Open a file dialog to select a .wav or .mp3 file."""
    file_path = filedialog.askopenfilename(title="Select an Audio File",
                                           filetypes=[("Audio Files", "*.wav *.mp3")])
    return file_path

def save_audio_to_csv(file_path, data_to_save='both'):
    """Reads an audio file and saves only amplitude data to a CSV file with a custom first line."""
    if not file_path:
        print("No file selected. Exiting.")
        return
    
    # Load the audio file
    data, sample_rate = sf.read(file_path)

    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Take the mean of stereo channels

    # Create time axis
    time = np.linspace(0, len(data) / sample_rate, num=len(data))

    # Define output CSV path
    output_csv = os.path.splitext(file_path)[0] + ".csv"

    if data_to_save == 'sound':
        with open(output_csv, "w") as f:
            f.write("% Amplitude\n")  # Custom first line
            np.savetxt(f, data, delimiter=",", fmt="%.6f")

    if data_to_save == 'both':
        with open(output_csv, "w") as f:
            f.write("% Time, Amplitude\n")  # Custom first line
            np.savetxt(f, np.column_stack((time, data)), delimiter=",", fmt="%.6f")

    print(f"CSV saved at: {output_csv}")

# Run the program
file_path = choose_audio_file()
# save_audio_to_csv(file_path, data_to_save='both')
save_audio_to_csv(file_path, data_to_save='sound')