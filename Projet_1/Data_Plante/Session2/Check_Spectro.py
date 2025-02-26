import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib

def get_wavelength_data(base_dir):
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Find the starting point of spectral data
                for i, line in enumerate(lines):
                    if ">>>>>Begin Spectral Data<<<<<" in line:
                        data_start = i + 1
                        break

                # Read spectral data and extract only wavelength values
                data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
                return np.array(data["Wavelength"].tolist())  # Return only the wavelength column as a list

def get_intensity_data_from_folder(base_dir):
    intensity_data = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(base_dir, file_name)

            with open(file_path, "r") as file:
                lines = file.readlines()
            
            # Find the starting point of spectral data
            for i, line in enumerate(lines):
                if ">>>>>Begin Spectral Data<<<<<" in line:
                    data_start = i + 1
                    break

            # Read spectral data and extract only intensity values
            data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
            intensity_data.append(data["Intensity"].tolist())  # Append only the intensity column as a list
    return intensity_data

def moving_average(data, window_size=50):
    kernel = np.ones(window_size) / window_size  # Create a uniform kernel
    return np.convolve(data, kernel, mode='same')  # Apply convolution


def Prepare_data(Plante_folder, Background_folder):
    Background_data = get_intensity_data_from_folder(Background_folder)
    Background_data = np.mean(Background_data, axis=0)
    Wavelenght_bins = get_wavelength_data(Background_folder)

    Spectro_data = get_intensity_data_from_folder(Plante_folder)
    Spectro_data = [arr for arr in Spectro_data if np.all(np.max(arr) <= 63500)] # Remove saturated

    # range de nanometre a analyser
    # Cap_low, Cap_high = 443, 657
    Cap_low, Cap_high = 505, 592

    low_index = np.argmin(np.abs(Wavelenght_bins - Cap_low))
    high_index = np.argmin(np.abs(Wavelenght_bins - Cap_high))

    # delta_bin = Wavelenght_bins[1]-Wavelenght_bins[0]
    # low_index = int((Cap_low - Wavelenght_bins[0])/delta_bin)
    # high_index = int((Cap_high - Wavelenght_bins[0])/delta_bin)

    Spectro_data =[arr[low_index:high_index] for arr in Spectro_data]
    Wavelenght_bins = Wavelenght_bins[low_index:high_index]
    Background_data = Background_data[low_index:high_index]

    Spectro_data = [moving_average(arr) for arr in Spectro_data]
    Background_data = moving_average(Background_data)
    Spectro_data = [(np.array(arr)/np.array(Background_data)) for arr in Spectro_data]
    Spectro_data = [arr/np.max(arr) for arr in Spectro_data]
    return Spectro_data, Wavelenght_bins

Plante_1_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Scindapsus_aureus_100ms\\"
Plante_2_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Kalanchoe_daigremontianum_100ms\\"
Background_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Background_30ms_feuille_blanche\\"

Plante_1_data, Wavelenght_bins = Prepare_data(Plante_1_folder, Background_folder)
Plante_2_data, _ = Prepare_data(Plante_2_folder, Background_folder)

plt.figure(figsize=(10, 5))
for i, data in enumerate(Plante_1_data):
    if i == 0:
        plt.plot(Wavelenght_bins, data, label='Scindapsus aureus', color = 'blue', linewidth=0.8)
    plt.plot(Wavelenght_bins, data, color = 'blue', linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectral Data Analysis by Plant Type")
    plt.grid()
plt.show()
for i, data in enumerate(Plante_2_data):
    if i == 0:
        plt.plot(Wavelenght_bins, data, label='Kalanchoe daigremontianum', color = 'red', linewidth=0.8)
    plt.plot(Wavelenght_bins, data, color = 'red', linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectral Data Analysis by Plant Type")
    plt.grid()
plt.show()












