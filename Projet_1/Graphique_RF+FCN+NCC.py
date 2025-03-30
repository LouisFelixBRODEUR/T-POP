import matplotlib.pyplot as plt

# Données RF, PCA+RF, PLS+RF
n_plantes = list(range(1, 21))
rf = [100.00, 98.14, 97.66, 96.88, 95.87, 95.15, 94.93, 94.47, 93.60, 93.34,
      93.12, 91.69, 91.72, 92.11, 91.80, 91.07, 90.81, 90.48, 90.33, 90.06]
pca_rf = [100.00, 98.52, 97.63, 97.10, 95.74, 95.03, 94.79, 93.65, 93.56, 93.70,
          92.89, 92.05, 92.00, 92.10, 91.75, 92.10, 91.62, 91.40, 90.93, 91.02]
pls_rf = [100.00, 99.35, 98.82, 98.58, 97.87, 97.53, 97.18, 96.73, 96.31, 96.20,
          96.06, 95.30, 95.22, 95.13, 95.23, 95.08, 94.82, 94.75, 94.72, 94.82]

# Nouvelles données FCN et NCC
accuracy_NN = [100.0, 98.154, 91.325, 87.268, 82.861, 81.841, 85.268, 86.253, 82.322, 82.646,
               84.962, 85.561, 84.814, 85.022, 85.184, 85.767, 87.808, 86.782, 86.381, 87.553]

accuracy_fit_mean = [100.0, 91.849, 88.022, 84.169, 82.075, 78.37, 77.876, 75.527, 72.425, 71.977,
                     72.443, 70.163, 70.703, 69.358, 68.729, 67.306, 66.798, 66.064, 65.329, 64.865]

# Création du graphique
plt.figure(figsize=(10, 6))

# Résultats précédents
plt.plot(n_plantes, rf, marker='o', linestyle='-', label='RF')
plt.plot(n_plantes, pca_rf, marker='o', linestyle='-', label='PCA+RF')
plt.plot(n_plantes, pls_rf, marker='o', linestyle='-', label='PLS+RF')

# Nouvelles données FCN et NCC
plt.plot(n_plantes, accuracy_NN, marker='o', linestyle='-', label='FCN')
plt.plot(n_plantes, accuracy_fit_mean, marker='o', linestyle='-', label='NCC')

# Paramétrage du graphique
plt.xlabel("Nombre de plantes dans l'ensemble de données", fontsize=16)
plt.ylabel("Précision (%)", fontsize=16)
plt.xticks(range(1, 21, 1), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', fontsize=14)
plt.grid(True)
plt.tight_layout()

# Afficher le graphique
plt.show()