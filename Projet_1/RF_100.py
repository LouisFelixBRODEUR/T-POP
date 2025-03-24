import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

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

# Nouvelle fonction pour uniformiser le nombre de scans par plante
def uniform_sample_scans(X, y, n_scans=100):
    """
    Pour chaque label, sélectionne aléatoirement au maximum n_scans scans.
    """
    unique_labels = np.unique(y)
    X_uniform_list = []
    y_uniform_list = []
    for label in unique_labels:
        indices = np.where(y == label)[0]
        if len(indices) > n_scans:
            selected_indices = np.random.choice(indices, n_scans, replace=False)
        else:
            selected_indices = indices
        X_uniform_list.append(X[selected_indices])
        y_uniform_list.append(y[selected_indices])
    X_uniform = np.concatenate(X_uniform_list, axis=0)
    y_uniform = np.concatenate(y_uniform_list, axis=0)
    return X_uniform, y_uniform

# Dossiers à charger
base_dir = os.path.dirname(os.path.abspath(__file__))
background_folder = os.path.join(base_dir, "Data_Plante", "Session3", "Background_7ms_feuille_blanche")
plante_folder_paths = [
    os.path.join(base_dir, "Data_Plante", "Session3", "Scindapsus_aureus_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session3", "Kalanchoe_daigremontianum_30ms"),
    os.path.join(base_dir, "Data_Plante", "Session3", "Dieffenbachia_seguine_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session3", "Dracaena_fragrans_10ms"),
    os.path.join(base_dir, "Data_Plante", "Session3", "Tradescantia_spathacea_top_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session3", "Tradescantia_spathacea_bot_25ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Euphorbia_milii_50ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Pachypodium_rosulatum_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Monstera_deliciosa_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Ficus_lyrata_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Begonia_gryphon_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Iresine_herbstii_50ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Spathiphyllum_cochlearispathum_35ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Philodendron_atabapoense_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Oldenlandia_affinis_20ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Dracaena_fragrans_30ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Dracaena_trifasciata_10ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Philodendron_melanochrysum_10ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Ficus_alii_40ms"),
    os.path.join(base_dir, "Data_Plante", "Session4", "Specialty_aglaonema_20ms")
]

# Chargement des données
X, y = load_data(plante_folder_paths, background_folder)

# Encodage des labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Uniformisation : garder au maximum 100 scans par plante
X_uniform, y_uniform = uniform_sample_scans(X, y_encoded, n_scans=100)

# Split des données sur les scans uniformisés
X_train, X_test, y_train, y_test = train_test_split(X_uniform, y_uniform, test_size=0.25, random_state=42, stratify=y_uniform)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluer_modele_rf_variants(X, y, nb_plantes, 
                               repetitions=5, 
                               test_size=0.25, 
                               n_components_pca=50, 
                               n_components_pls=30,
                               random_state=60):
    """
    Sélectionne aléatoirement nb_plantes parmi les labels uniques de y,
    filtre X et y pour ces plantes, sépare en train/test, entraîne et
    évalue 3 variantes de Random Forest (simple, PCA, PLS).
    
    Répète ce processus 'repetitions' fois pour faire une moyenne.
    
    Paramètres
    ----------
    X : ndarray de forme (N, d)
        Les données spectrales (ou autres features).
    y : ndarray de forme (N,)
        Les labels (entiers) encodés.
    nb_plantes : int
        Nombre de plantes (labels) à sélectionner pour cette évaluation.
    repetitions : int
        Nombre de répétitions pour la moyenne.
    test_size : float
        Proportion de l'ensemble retenue pour le test (ex: 0.25).
    n_components_pca : int
        Nombre de composantes principales à garder pour PCA.
    n_components_pls : int
        Nombre de composantes latentes à garder pour PLS.
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Renvoie
    -------
    (acc_rf_moy, acc_pca_moy, acc_pls_moy) : tuple de float
        Les précisions moyennes (accuracy) sur 'repetitions' essais
        pour chaque variante (RF simple, RF+PCA, RF+PLS).
    """
    
    rng = np.random.default_rng(random_state)
    labels_uniques = np.unique(y)

    acc_rf_list = []
    acc_pca_list = []
    acc_pls_list = []
    
    for rep in range(repetitions):
        # 1) Sélection aléatoire de nb_plantes parmi les labels disponibles
        plantes_choisies = rng.choice(labels_uniques, size=nb_plantes, replace=False)
        
        # 2) Filtrage des données pour ne garder que ces plantes
        mask = np.isin(y, plantes_choisies)
        X_sub, y_sub = X[mask], y[mask]
        
        # 3) Mélange + séparation train/test
        X_sub, y_sub = shuffle(X_sub, y_sub, random_state=rng.integers(1_000_000))
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub,
            test_size=test_size,
            stratify=y_sub,
            random_state=rng.integers(1_000_000)
        )
        
        # Prétraitement commun (scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        
        # 1) Random Forest (simple)
        rf = RandomForestClassifier(random_state=rng.integers(1_000_000))
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        # 2) PCA + Random Forest
        pca = PCA(n_components=n_components_pca, random_state=rng.integers(1_000_000))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)
        
        rf_pca = RandomForestClassifier(random_state=rng.integers(1_000_000))
        rf_pca.fit(X_train_pca, y_train)
        y_pred_pca = rf_pca.predict(X_test_pca)
        acc_pca = accuracy_score(y_test, y_pred_pca)
        
        # 3) PLS + Random Forest
        pls = PLSRegression(n_components=n_components_pls)
        pls.fit(X_train_scaled, y_train)
        X_train_pls = pls.transform(X_train_scaled)
        X_test_pls  = pls.transform(X_test_scaled)
        
        rf_pls = RandomForestClassifier(random_state=rng.integers(1_000_000))
        rf_pls.fit(X_train_pls, y_train)
        y_pred_pls_rf = rf_pls.predict(X_test_pls)
        acc_pls = accuracy_score(y_test, y_pred_pls_rf)
        
        acc_rf_list.append(acc_rf)
        acc_pca_list.append(acc_pca)
        acc_pls_list.append(acc_pls)
    
    return (
        np.mean(acc_rf_list), 
        np.mean(acc_pca_list), 
        np.mean(acc_pls_list)
    )

def graphique_precision_rf_variants(X, y, max_plantes, repetitions=5, test_size=0.25):
    """
    Trace la précision (accuracy) de 3 variantes de Random Forest 
    (RF simple, RF+PCA, RF+PLS) en fonction du nombre de plantes (classes) 
    utilisé dans le dataset.
    
    Paramètres
    ----------
    X : ndarray (N, d)
        Données complètes.
    y : ndarray (N,)
        Labels encodés.
    max_plantes : int
        Nombre maximal de classes à tester (de 1 à max_plantes).
    repetitions : int
        Nombre de répétitions pour chaque nombre de plantes.
    test_size : float
        Proportion test dans le split (0.25 par défaut).
    """
    
    rf_accuracies   = []
    pca_accuracies  = []
    pls_accuracies  = []
    
    for n in range(1, max_plantes + 1):
        acc_rf, acc_pca, acc_pls = evaluer_modele_rf_variants(
            X, y, nb_plantes=n,
            repetitions=repetitions,
            test_size=test_size,
            n_components_pca=10,
            n_components_pls=10,
        )
        rf_accuracies.append(acc_rf * 100)
        pca_accuracies.append(acc_pca * 100)
        pls_accuracies.append(acc_pls * 100)
        
        print(f"n={n} plantes -> RF={acc_rf*100:.2f}%, PCA+RF={acc_pca*100:.2f}%, PLS+RF={acc_pls*100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_plantes+1), rf_accuracies, marker='o', label='Random Forest')
    plt.plot(range(1, max_plantes+1), pca_accuracies, marker='o', label='RF + PCA')
    plt.plot(range(1, max_plantes+1), pls_accuracies, marker='o', label='RF + PLS')
    
    plt.title('Précision en fonction du nombre de plantes (classes)')
    plt.xlabel('Nombre de plantes (classes) sélectionnées')
    plt.ylabel('Précision moyenne (%)')
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Appel final : tester jusqu’à 20 plantes avec 5 répétitions pour chaque configuration.
max_plantes = 20  
repetitions = 5

graphique_precision_rf_variants(X, y_encoded, max_plantes=max_plantes, repetitions=repetitions)
