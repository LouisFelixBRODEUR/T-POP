import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib

# Identifie les folders qui commence avec 'Plante'
def get_Plante_folders(base_dir):
    Plante_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Plante")]
    return Plante_folders

# Identifie les folders qui commence avec 'Tige' et 'Feuille'
def get_tige_feuille_folders(base_dir):
    feuille_folders = []
    tige_folders = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path):
            if d.startswith("Feuille_"):
                feuille_folders.append(full_path)
            elif d.startswith("Tige_"):
                tige_folders.append(full_path)
    return feuille_folders, tige_folders

def get_wavelength_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the starting point of spectral data
    for i, line in enumerate(lines):
        if ">>>>>Begin Spectral Data<<<<<" in line:
            data_start = i + 1
            break

    # Read spectral data and extract only wavelength values
    data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
    return data["Wavelength"].tolist()  # Return only the wavelength column as a list

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

def make_plant_Database():
    Plant_DataBase = {}
    for Plante_Folder in get_Plante_folders(os.path.dirname(os.path.abspath(__file__))):
        Plante_name = Plante_Folder.split('\\')[-1]
        Plant_DataBase[Plante_name] = {'Feuille':{}, 'Tige':{}}
        tige_plante_path = get_tige_feuille_folders(Plante_Folder)

        for feuille_number_path in tige_plante_path[0]:
            Plant_DataBase[Plante_name]['Feuille'][feuille_number_path.split('\\')[-1]] = get_intensity_data_from_folder(feuille_number_path)
        
        for tige_number_path in tige_plante_path[1]:
            Plant_DataBase[Plante_name]['Tige'][tige_number_path.split('\\')[-1]] = get_intensity_data_from_folder(tige_number_path)

    for file_name in os.listdir(tige_number_path):
        if file_name.endswith(".txt"):
            wavelength_path = os.path.join(tige_number_path, file_name)
    Plant_DataBase['wavelength'] = get_wavelength_data(wavelength_path)

    return Plant_DataBase

def moving_average(data, window_size=50):
    kernel = np.ones(window_size) / window_size  # Create a uniform kernel
    return np.convolve(data, kernel, mode='same')  # Apply convolution

def get_background():
    light_source_background_path = os.path.dirname(os.path.abspath(__file__))+"\\BackGround\\"
    return get_intensity_data_from_folder(light_source_background_path)[0]

def Show_data(Plant_DataBase, plant_part_to_show):
    Background_data = moving_average(get_background())
    wavelength = Plant_DataBase['wavelength']

    # range de nanometre a analyser
    Cap_high = 675
    Cap_low = 450
    wavelength_0 = wavelength[0]
    wavelength_delta_bin = wavelength[1]-wavelength[0]
    low_index = int((Cap_low - wavelength_0)/wavelength_delta_bin)
    high_index = int((Cap_high - wavelength_0)/wavelength_delta_bin)
    wavelength = wavelength[low_index:high_index]
    Background_data = Background_data[low_index:high_index]

    plant_types = list(Plant_DataBase.keys())[:-1]
    # color_map = plt.get_cmap('tab10', len(plant_types))
    # plant_colors = {plant: color_map(i) for i, plant in enumerate(plant_types)}
    plant_colors = ['red', 'blue']
    plt.figure(figsize=(10, 5))
    for j, Plante_type in enumerate(plant_types):
        color = plant_colors[j]  # Get assigned color for the plant type
        for plant_part in Plant_DataBase[Plante_type].keys():
            if plant_part == plant_part_to_show:
                for plant_part_number in Plant_DataBase[Plante_type][plant_part].keys():
                    label = f'{Plante_type} {plant_part_number}'
                    for i, data in enumerate(Plant_DataBase[Plante_type][plant_part][plant_part_number]):
                        data = moving_average(data[low_index:high_index])
                        data = (np.array(data)/np.array(Background_data))
                        data = data/np.max(data)
                        plt.plot(wavelength, data, label=label + f'.{i+1}', color=color, linewidth=0.8)
    # plt.plot(wavelength, Background_data, label='LS', color='red', linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectral Data Analysis by Plant Type")
    plt.legend()
    plt.grid()
    plt.show()

Database = make_plant_Database()
Show_data(Database, 'Feuille')
Show_data(Database, 'Tige')




