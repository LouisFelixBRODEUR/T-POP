import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions d'importation et préparation des données (inchangées)
def get_wavelength_data(base_dir):
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
                data_start = next(i + 1 for i, line in enumerate(lines) if ">>>>>Begin Spectral Data<<<<<" in line)
                data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
                return np.array(data["Wavelength"].tolist())

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
            data_start = next(i + 1 for i, line in enumerate(lines) if ">>>>>Begin Spectral Data<<<<<" in line)
            data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
            intensity_data.append(data["Intensity"].tolist())
    return intensity_data

def Prepare_data(Plante_folder, Background_folder):
    bg_data = get_intensity_data_from_folder(Background_folder)
    Background_data = np.mean(bg_data, axis=0)
    Wavelength_bins = get_wavelength_data(Background_folder)
    spectro_data = get_intensity_data_from_folder(Plante_folder)
    Spectro_data = [arr for arr in spectro_data if np.all(np.max(arr) <= 63500)]
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

# (Tes chemins vers dossiers de plantes ici, inchangés...)
plante_folder_paths = [ # Mets tes chemins réels ici
    os.path.join(base_dir, "Data_Plante", "Session3", "Scindapsus_aureus_20ms"),
    # Ajoute tous tes dossiers ici...
]

# Chargement et préparation des données
X, y = load_data(plante_folder_paths, background_folder)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Random Forest (original)
rf = RandomForestClassifier(random_state=70)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# PCA + Random Forest
rf_pca = RandomForestClassifier(random_state=60)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)

# PLS + Random Forest
pls = PLSRegression(n_components=30)
pls.fit(X_train_scaled, y_train)
X_train_pls = pls.transform(X_train_scaled)
X_test_pls = pls.transform(X_test_scaled)
rf_pls = RandomForestClassifier(random_state=30)
rf_pls.fit(X_train_pls, y_train)
y_pred_pls_rf = rf_pls.predict(X_test_pls)

# Ajout de GridSearchCV (Nouveau modèle optimisé)
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [10, 20, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 5]}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_scaled)

# Résultats
results = pd.DataFrame({
    'Modèle': ['Random Forest', 'PCA + Random Forest', 'PLS + Random Forest', 'Optimized RF (GridSearchCV)'],
    'Accuracy (%)': [
        accuracy_score(y_test, y_pred_rf) * 100,
        accuracy_score(y_test, y_pred_pca) * 100,
        accuracy_score(y_test, y_pred_pls_rf) * 100,
        accuracy_score(y_test, y_pred_best_rf) * 100
    ]
})

print(results)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Modèle', y='Accuracy (%)', data=results)
plt.title('Comparaison des modèles avec/sans GridSearchCV')
plt.ylim(0, 100)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
