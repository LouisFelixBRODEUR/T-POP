import os
import pandas as pd
import matplotlib.pyplot as plt

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





path_feuille = os.path.dirname(os.path.abspath(__file__))+"\\Plante_Dragon\\USB4F104151__0__12-54-15-542.txt"
# path_tige = os.path.dirname(os.path.abspath(__file__))+"\\Plante_Limace\\Fente_0_08.csv"
# path = os.path.dirname(os.path.abspath(__file__))+"\\Plante_Dragon\\USB4F104151__1__12-55-13-447.txt"


# path_background = os.path.dirname(os.path.abspath(__file__))+"\\BackGround\\Fente_0_08.csv"

# file_path = "USB4F104151__19__12-14-57-321.txt"

# Read the file
with open(path_feuille, "r") as file:
    lines = file.readlines()

# Find the line where spectral data begins
for i, line in enumerate(lines):
    if ">>>>>Begin Spectral Data<<<<<" in line:
        data_start = i + 1
        break

# Read spectral data into a DataFrame
data = pd.read_csv(path_feuille, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])

# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(data["Wavelength"], data["Intensity"], label="Spectrum", color='b')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Spectral Data Analysis")
plt.legend()
plt.grid()
plt.show()
