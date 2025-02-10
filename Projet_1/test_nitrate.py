import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Simulation des données spectrales
# -------------------------------

# Définition des longueurs d'onde (200 à 220 nm)
wavelengths = np.linspace(200, 220, 101)  # 101 points de 200 nm à 220 nm
optimal_wavelength = 210  # Longueur d'onde optimale pour l'absorption des nitrates
sigma = 2               # Écart-type de la distribution gaussienne
amplitude = 0.001       # Facteur d'échelle (absorbance par mg/L)
baseline = 0.05         # Absorbance de base (sans présence de nitrate)

# Nombre d'échantillons pour la calibration
n_samples = 30

# Génération aléatoire de concentrations de nitrate (en mg/L)
np.random.seed(42)
concentrations = np.random.uniform(0, 1000, n_samples)  # Valeurs comprises entre 0 et 1000 mg/L

def simulate_spectrum(concentration, wavelengths, optimal_wavelength, sigma, amplitude, baseline):
    """
    Simule le spectre d'absorption d'une solution de nitrate.
    L'absorbance à chaque longueur d'onde est modélisée par une distribution gaussienne centrée
    sur la longueur d'onde optimale et proportionnelle à la concentration.
    Un bruit aléatoire est ajouté pour simuler la variabilité expérimentale.
    """
    spectrum = baseline + (amplitude * concentration) * np.exp(-((wavelengths - optimal_wavelength)**2) / (2 * sigma**2))
    noise = np.random.normal(0, 0.002, size=wavelengths.shape)  # Bruit gaussien (écart-type 0.002)
    return spectrum + noise

# Génération de la matrice X des spectres et du vecteur y des concentrations
X = np.array([simulate_spectrum(c, wavelengths, optimal_wavelength, sigma, amplitude, baseline)
              for c in concentrations])
y = concentrations

# Affichage de quelques spectres simulés
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.plot(wavelengths, X[i, :], label=f'Concentration: {y[i]:.1f} mg/L')
plt.xlabel('Longueur d\'onde (nm)')
plt.ylabel('Absorbance')
plt.title('Spectres d\'absorption simulés')
plt.legend()
plt.show()

# -------------------------------
# 2. Application de la régression PLSR
# -------------------------------

# Division de l'ensemble de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition du nombre de composantes latentes à utiliser (par exemple 2)
n_components = 2
pls = PLSRegression(n_components=n_components)

# Ajustement du modèle sur les données d'entraînement
pls.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = pls.predict(X_test).ravel()

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Erreur quadratique moyenne (MSE) sur l'ensemble test :", mse)
print("Coefficient de détermination (R^2) sur l'ensemble test :", r2)

# Validation croisée (avec 5 folds)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pls, X, y, cv=cv, scoring='r2')
print("Scores R^2 en validation croisée :", cv_scores)
print("Score R^2 moyen en validation croisée :", np.mean(cv_scores))

# Visualisation des concentrations prédites vs. vraies
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Parfait')
plt.xlabel('Concentration réelle (mg/L)')
plt.ylabel('Concentration prédite (mg/L)')
plt.title('PLSR : Concentations réelles vs. prédite')
plt.legend()
plt.show()

# -------------------------------
# 3. Prédiction sur un nouvel échantillon
# -------------------------------

# Simulation d'un nouveau spectre pour une solution ayant par exemple 750 mg/L de nitrate
new_concentration = 750  # Valeur réelle pour le test
new_spectrum = simulate_spectrum(new_concentration, wavelengths, optimal_wavelength, sigma, amplitude, baseline)
new_spectrum = new_spectrum.reshape(1, -1)  # Mise en forme pour la prédiction

# Prédiction à l'aide du modèle PLSR
predicted_new_concentration = pls.predict(new_spectrum).ravel()[0]
print(f"Nouvelle concentration réelle : {new_concentration} mg/L, prédite : {predicted_new_concentration:.1f} mg/L")
