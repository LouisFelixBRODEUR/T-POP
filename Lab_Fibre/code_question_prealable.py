import numpy as np
import matplotlib.pyplot as plt

# Définition des constantes 
lambda_0 = 632.8e-9  # Longueur d'onde du laser en mètres (632.8 nm)
w_laser = 315e-6  # Taille du faisceau à la sortie du laser en mètres (315 µm)
f = 4.5e-3  # Distance focale de la lentille en mètres (4.5 mm)

# Caractéristiques de la fibre monomode
NA = 0.12  # Ouverture numérique
lambda_c = 620e-9  # Longueur d'onde de coupure en mètres (620 nm)
V_c = 2.405  # Valeur critique de V pour une fibre monomode

# Calcul du rayon du coeur de la fibre
a = (V_c * lambda_c) / (2 * np.pi * NA)

# Définition de la plage de distances Z 
Z = np.linspace(0, 0.5, 1000)  # Distance de 0 à 50 cm 

# Calcul de l'angle de divergence 
theta = lambda_0 / (np.pi * w_laser)

# Calcul de la taille du faisceau au niveau de la lentille 
w_objectif = w_laser * np.sqrt(1 + ((Z * np.tan(theta)) / w_laser) ** 2)

# Calcul de la taille du faisceau après la lentille 
w_image = (lambda_0 * f) / (np.pi * w_objectif)

# Calcul de la taille du mode fondamental de la fibre 
w_0=a * (0.65 + 1.619 / (V_c ** 1.5) + 2.879 / (V_c ** 6))

# Calcul de l'efficacité de couplage T 
T = ((2 * w_image * w_0)/ (w_image** 2 + w_0 ** 2)) ** 2

# Trouver la valeur de Z qui maximise T
Z_optimal_index = np.argmax(T)  # Index du maximum
Z_optimal = Z[Z_optimal_index]  # Valeur de Z optimale


# Trouver la frequence normalisee 
V =((2 * np.pi * a)/ lambda_0) * NA
print(V)

# Affichage de la valeur optimale de Z
print(f"Valeur optimale de Z pour maximiser l'efficacité de couplage : {Z_optimal*100:.3f} cm")

# Tracé de la courbe T(Z) avec la valeur exacte de w_laser
plt.figure(figsize=(8, 6))
plt.plot(Z * 100, T, label="Efficacité d'injection T", color="red")
plt.axvline(Z_optimal * 100, color='black', linestyle="--", label=f"Z optimal = {Z_optimal*100:.3f} cm")
plt.xlabel("Distance Z (cm)")
plt.ylabel("Efficacité de couplage T")
plt.title("Efficacité de couplage en fonction de la distance Z")
plt.grid(True)
plt.legend()
plt.show()



