import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle

# Fonctions originales pour importer et préparer tes données
def get_wavelength_data(base_dir):
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if ">>>>>Begin Spectral Data<<<<<" in line:
                        data_start = i + 1
                        break
                data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
                return np.array(data["Wavelength"])

def moving_average(data, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def get_intensity_data_from_folder(base_dir):
    intensity_data = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):
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
    Background_data = np.mean(bg_data, axis=0)
    Wavelength_bins = get_wavelength_data(Background_folder)

    spectro_data = get_intensity_data_from_folder(Plante_folder)
    Spectro_data = [arr for arr in spectro_data if np.max(arr) <= 63500]

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

# Calcul des références moyennes par plante
def calculer_references(X, y):
    references = {}
    for label in np.unique(y):
        references[label] = np.mean(X[y == label], axis=0)
    return references

# Classification par corrélation
def classifier_par_correlation(x_sample, references):
    correlations = {label: np.corrcoef(x_sample, ref)[0, 1] for label, ref in references.items()}
    return max(correlations, key=correlations.get)

# Chargement des données
base_dir = os.path.dirname(os.path.abspath(__file__))
background_folder = os.path.join(base_dir, "Data_Plante", "Session2", "Background_30ms_feuille_blanche")
plante_folder_paths = [
    os.path.join(base_dir,"Data_Plante", "Session2", "Kalanchoe_daigremontianum_100ms"),
    os.path.join(base_dir,"Data_Plante", "Session2", "Scindapsus_aureus_100ms")
    
]

X, y = load_data(plante_folder_paths, background_folder)

# Encodage des labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Évaluation du modèle avec visualisation
def evaluer_modele(X, y, nb_plantes, repetitions=10, test_size=0.25, afficher_confusion=True):
    scores = []
    labels_uniques = np.unique(y)

    for _ in range(repetitions):
        plantes_choisies = np.random.choice(labels_uniques, nb_plantes, replace=False)
        indices_selectionnes = np.isin(y, plantes_choisies)

        X_selected, y_selected = shuffle(X[indices_selectionnes], y[indices_selectionnes])
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, stratify=y_selected)

        references = calculer_references(X_train, y_train)
        predictions = [classifier_par_correlation(sample, references) for sample in X_test]

        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)

        if afficher_confusion:
            cm = confusion_matrix(y_test, predictions, labels=plantes_choisies)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=plantes_choisies)
            disp.plot(cmap='viridis')
            plt.title('Matrice de confusion')
            plt.show()

    accuracy_moyenne = np.mean(scores)
    accuracy_std = np.std(scores)
    print(f"Accuracy moyenne: {accuracy_moyenne:.2f} ± {accuracy_std:.2f}")
    
    return accuracy_moyenne, accuracy_std

# Graphique de précision selon le nombre de plantes
def graphique_precision(X, y, max_plantes, repetitions=5):
    nb_plantes = range(1, max_plantes + 1)
    precisions = []

    for n in nb_plantes:
        # On désactive l'affichage de la matrice de confusion pour éviter la multiplication des graphiques
        acc, _ = evaluer_modele(X, y, nb_plantes=n, repetitions=repetitions, afficher_confusion=False)
        precisions.append(acc * 100)

    plt.plot(list(nb_plantes), precisions, marker='o')
    plt.xlabel('Nombre de plantes dans le dataset')
    plt.ylabel('Précision (%)')
    plt.title('Précision selon le nombre de plantes')
    plt.grid()
    plt.show()

# Exemple d'utilisation :
# Affichage des matrices de confusion (5 graphiques)
# accuracy_moyenne, accuracy_std = evaluer_modele(X_scaled, y_encoded, nb_plantes=20, repetitions=10, afficher_confusion=True)

# Affichage du graphique de précision (sans matrices de confusion supplémentaires)
graphique_precision(X_scaled, y_encoded, max_plantes=2, repetitions=20)