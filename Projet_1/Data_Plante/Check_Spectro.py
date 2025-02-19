import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

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

# c:\Users\louis\Documents\ULaval_S4\TPOP\GitWorkSpace\Projet_1\Data_Plante\Plante_Dragon
# c:\\Users\\louis\\Documents\\ULaval_S4\\TPOP\\GitWorkSpace\\Projet_1\\Data_Plante\\Plante_Dragon

# donne une liste de couple de list pour le data dans le folder [(x1,y1),(x2,y2),(x3,y3),...]
def get_data_from_folder(base_dir):
    spectral_data = {}
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                if ">>>>>Begin Spectral Data<<<<<" in line:
                    data_start = i + 1
                    break
            data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
            spectral_data[file_name] = list(data.itertuples(index=False, name=None))

    return spectral_data





# path_feuille_drag = os.path.dirname(os.path.abspath(__file__))+"\\Plante_Dragon\\Feuille_1\\USB4F104151__19__12-14-57-321.txt"
# path_feuille_lim = os.path.dirname(os.path.abspath(__file__))+"\\Plante_Limace(Golden_Pothos)\\Feuille_1\\USB4F104151__0__11-38-47-480.txt"

# # Read the file
# def data_from(path):
#     with open(path, "r") as file:
#         lines = file.readlines()
#     for i, line in enumerate(lines):
#         if ">>>>>Begin Spectral Data<<<<<" in line:
#             data_start = i + 1
#             break
#     return pd.read_csv(path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])

# data_lim = data_from(path_feuille_lim)
# data_drag = data_from(path_feuille_drag)
# longueur_donde_lim = data_lim["Wavelength"]
# longueur_donde_drag = data_drag["Wavelength"]
# intensite_lim = data_lim["Intensity"]
# intensite_drag = data_drag["Intensity"]

# intensite_lim = intensite_lim/np.max(intensite_lim)
# intensite_drag = intensite_drag/np.max(intensite_drag)

Plant_DataBase = {}
for Plante_Folder in get_Plante_folders(os.path.abspath(__file__)):
    Plant_DataBase[Plante_Folder] = {}





# # Plot the spectrum
# plt.figure(figsize=(10, 5))
# plt.plot(longueur_donde_lim, intensite_lim, label="Limace", color='b')
# plt.plot(longueur_donde_drag, intensite_drag, label="Dragon", color='r')
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity")
# plt.title("Spectral Data Analysis")
# plt.legend()
# plt.grid()
# plt.show()
