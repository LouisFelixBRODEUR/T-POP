import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions pour importer et préparer tes données
def get_wavelength_data(base_dir):
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt") and os.path.isfile(os.path.join(base_dir, file_name)):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if ">>>>>Begin Spectral Data<<<<<" in line:
                        data_start = i + 1
                        break
                data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
                return np.array(data["Wavelength"].tolist())

def moving_average(data, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def get_intensity_data_from_folder(base_dir):
    intensity_data = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt") and os.path.isfile(os.path.join(base_dir, file_name)):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                if ">>>>>Begin Spectral Data<<<<<" in line:
                    data_start = i + 1
                    break
            data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
            intensity_data.append(data["Intensity"].tolist())
    return intensity_data

def Prepare_data(Plante_folder, Background_folder):
    bg_data = get_intensity_data_from_folder(Background_folder)
    if not bg_data:
        raise FileNotFoundError(f"Aucun fichier de fond trouvé dans {Background_folder}")
    Background_data = np.mean(bg_data, axis=0)
    Wavelength_bins = get_wavelength_data(Background_folder)

    spectro_data = get_intensity_data_from_folder(Plante_folder)
    if not spectro_data:
        raise FileNotFoundError(f"Aucun fichier de spectre trouvé dans {Plante_folder}")
    Spectro_data = spectro_data
    Spectro_data = [arr for arr in Spectro_data if np.all(np.max(arr) <= 63500)]

    Cap_low, Cap_high = 420, 800
    low_index = np.argmin(np.abs(Wavelength_bins - Cap_low))
    high_index = np.argmin(np.abs(Wavelength_bins - Cap_high))

    Background_data = moving_average(Background_data, window_size=5)
    Spectro_data = [moving_average(arr / Background_data, window_size=30)[low_index:high_index] for arr in Spectro_data]
    Spectro_data = [arr / np.max(arr) for arr in Spectro_data]

    return Spectro_data

def load_data(plante_folder_paths, background_folder):
    data, labels = [], []
    for path in plante_folder_paths:
        prepared_data = Prepare_data(path, background_folder)
        data.extend(prepared_data)
        labels.extend([os.path.basename(path)] * len(prepared_data))
    return np.array(data), np.array(labels)

# Dossiers à charger
base_dir = os.path.dirname(os.path.abspath(__file__))
background_folder = os.path.join(base_dir, "Data_Plante", "Session3", "Background_7ms_feuille_blanche")
plante_folder_paths = [
    os.path.join(base_dir,"Data_Plante", "Session3", "Scindapsus_aureus_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Kalanchoe_daigremontianum_30ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Dieffenbachia_seguine_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Dracaena_fragrans_10ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Tradescantia_spathacea_top_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante1_50ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante2_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante3_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante4_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante5_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante6_50ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante7_35ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante8_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante9_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante10_30ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante11_10ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante12_10ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante13_40ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Plante14_20ms")


]

# Chargement des données
X, y = load_data(plante_folder_paths, background_folder)

# Encodage des labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir des plages de valeurs pour n_components et random_state
n_components_values = np.linspace(10, 100, num=10, endpoint=True, dtype=int)
rf_states = np.linspace(10, 100, num=10, endpoint=True, dtype=int)

# Initialisation pour stocker les meilleurs scores et paramètres

best_score_rf = 0
best_params_rf = {}

best_score_pca = 0
best_params_pca = {}

best_score_pls = 0
best_params_pls = {}

# Recherche pour Random Forest sur les données standardisées (sans réduction)
for rs in rf_states:
    rf = RandomForestClassifier(random_state=rs)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    
    if score > best_score_rf:
        best_score_rf = score
        best_params_rf = {'random_state': rs}

# Recherche pour la pipeline PCA + Random Forest
for n in n_components_values:
    # Réduction de dimension avec PCA
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    for rs in rf_states:
        # Entraînement d'un Random Forest avec random_state = rs sur les données transformées par PCA
        rf_pca = RandomForestClassifier(random_state=rs)
        rf_pca.fit(X_train_pca, y_train)
        y_pred = rf_pca.predict(X_test_pca)
        score = accuracy_score(y_test, y_pred)
        
        # Mise à jour si on trouve un meilleur score
        if score > best_score_pca:
            best_score_pca = score
            best_params_pca = {'n_components': n, 'random_state': rs}



# Recherche pour la pipeline PLS + Random Forest
for n in n_components_values:
    # Réduction de dimension avec PLS
    pls = PLSRegression(n_components=n)
    pls.fit(X_train_scaled, y_train)
    X_train_pls = pls.transform(X_train_scaled)
    X_test_pls = pls.transform(X_test_scaled)
    
    for rs in rf_states:
        # Entraînement d'un Random Forest avec random_state = rs sur les données transformées par PLS
        rf_pls = RandomForestClassifier(random_state=rs)
        rf_pls.fit(X_train_pls, y_train)
        y_pred = rf_pls.predict(X_test_pls)
        score = accuracy_score(y_test, y_pred)
        
        # Mise à jour si on trouve un meilleur score
        if score > best_score_pls:
            best_score_pls = score
            best_params_pls = {'n_components': n, 'random_state': rs}

# Affichage des résultats optimaux pour chaque pipeline
print("Meilleur score pour RF sur données standardisées: {:.2f}% avec random_state = {}"
      .format(best_score_rf * 100, best_params_rf['random_state']))

print("Meilleur score pour PCA + RF: {:.2f}% avec n_components = {} et random_state = {}"
      .format(best_score_pca * 100, best_params_pca['n_components'], best_params_pca['random_state']))

print("Meilleur score pour PLS + RF: {:.2f}% avec n_components = {} et random_state = {}"
      .format(best_score_pls * 100, best_params_pls['n_components'], best_params_pls['random_state']))
