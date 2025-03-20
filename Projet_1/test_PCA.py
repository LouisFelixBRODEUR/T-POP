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
# plante_folder_paths = [
#     os.path.join(base_dir,"Data_Plante", "Session3", "Scindapsus_aureus_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session3", "Kalanchoe_daigremontianum_30ms"),
#     os.path.join(base_dir,"Data_Plante", "Session3", "Dieffenbachia_seguine_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session3", "Dracaena_fragrans_10ms"),
#     os.path.join(base_dir,"Data_Plante", "Session3", "Tradescantia_spathacea_top_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante1_50ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante2_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante3_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante4_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante5_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante6_50ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante7_35ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante8_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante9_20ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante10_30ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante11_10ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante12_10ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante13_40ms"),
#     os.path.join(base_dir,"Data_Plante", "Session4", "Plante14_20ms")
# ]
plante_folder_paths = [
    os.path.join(base_dir,"Data_Plante", "Session3", "Scindapsus_aureus_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Kalanchoe_daigremontianum_30ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Dieffenbachia_seguine_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Dracaena_fragrans_10ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Tradescantia_spathacea_top_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session3", "Tradescantia_spathacea_bot_25ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Euphorbia_milii_50ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Pachypodium_rosulatum_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Monstera_deliciosa_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Ficus_lyrata_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Begonia_gryphon_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Iresine_herbstii_50ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Spathiphyllum_cochlearispathum_35ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Philodendron_atabapoense_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Oldenlandia_affinis_20ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Dracaena_fragrans_30ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Dracaena_trifasciata_10ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Philodendron_melanochrysum_10ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Ficus_alii_40ms"),
    os.path.join(base_dir,"Data_Plante", "Session4", "Specialty_aglaonema_20ms")
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


# PLSR utilisé directement pour la classification (en arrondissant les prédictions)
# plsr = PLSRegression(n_components=70)
# plsr.fit(X_train_scaled, y_train)
# y_pred_plsr = plsr.predict(X_test_scaled)
# y_pred_plsr_classes = np.round(y_pred_plsr).astype(int).flatten()
# y_pred_plsr_classes = np.clip(y_pred_plsr_classes, 0, len(le.classes_)-1)

# PLS + Random Forest : utilisation du PLS pour extraire des composantes supervisées puis Random Forest pour la classification
pls = PLSRegression(n_components=30)
pls.fit(X_train_scaled, y_train)
X_train_pls = pls.transform(X_train_scaled)
X_test_pls = pls.transform(X_test_scaled)
# # Prédictions sur les données de test
# y_pred_pls = pls.predict(X_test_scaled)

# # # Si c'est un problème de classification, tu peux arrondir les prédictions pour obtenir des classes discrètes
# y_pred_pls_classes = np.round(y_pred_pls).astype(int)

# Obtenir R² sur les données d'entraînement
r2_train = pls.score(X_train_scaled, y_train)
print(f"R² sur les données d'entraînement : {r2_train:.2f}")

# Obtenir R² sur les données de test
r2_test = pls.score(X_test_scaled, y_test)
print(f"R² sur les données de test : {r2_test:.2f}")


rf_pls = RandomForestClassifier(random_state=30)
rf_pls.fit(X_train_pls, y_train)
y_pred_pls_rf = rf_pls.predict(X_test_pls)




# Résultats
results = pd.DataFrame({
    'Modèle': ['Random Forest', 'PCA + Random Forest', 'PLS + Random Forest'],
    'Accuracy (%)': [
        accuracy_score(y_test, y_pred_rf) * 100,
        accuracy_score(y_test, y_pred_pca) * 100,
        accuracy_score(y_test, y_pred_pls_rf) * 100
        
    ]
})

print(results)


# Calculer la moyenne et l'écart-type des données standardisées
print("Moyenne des données après standardisation :")
print(np.mean(X_train_scaled, axis=0))  # Devrait être proche de 0
print("\nÉcart-type des données après standardisation :")
print(np.std(X_train_scaled, axis=0))  # Devrait être proche de 1


# # Visualisation
# # Graphique de la variance expliquée par PCA
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# # Graphique des scores de PLS
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pls[:, 0], X_train_pls[:, 1], c=y_train, cmap='viridis')
plt.title('PLS Regression Scores Plot')
plt.xlabel('PLS1')
plt.ylabel('PLS2')
plt.colorbar(label='Class Labels')
plt.tight_layout()
plt.show()

# Graphique de l'importance des caractéristiques dans Random Forest
plt.figure(figsize=(10, 6))

# # Calcul de l'importance des caractéristiques
feature_importances = rf.feature_importances_

# # Tri des caractéristiques par importance
indices = np.argsort(feature_importances)[::-1]

# # Nombre de caractéristiques à afficher (par exemple, les 20 plus importantes)
n_top_features = 20

# # Affichage du graphique pour les 20 caractéristiques les plus importantes
plt.bar(range(n_top_features), feature_importances[indices][:n_top_features])

# # Affichage des longueurs d'onde correspondantes aux caractéristiques les plus importantes
wavelengths = np.linspace(420, 800, X_train_scaled.shape[1])  # Ajuste cela selon tes données
plt.xticks(range(n_top_features), [f'{wavelength:.0f} nm' for wavelength in wavelengths[indices][:n_top_features]], rotation=90)

plt.title('Top 20 Feature Importances in Random Forest')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()


# # Graphique de comparaison des résultats de classification
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Modèle', y='Accuracy (%)', data=results, errorbar=None)
plt.title('Comparaison des résultats de classification')
plt.xlabel("Modèle de classification")
plt.ylabel("Précision (%)")
plt.ylim(0, 100)
plt.tight_layout()

# # Ajouter des annotations pour afficher la valeur de précision au-dessus de chaque barre
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.3f}%', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=12, color='black')

plt.show()