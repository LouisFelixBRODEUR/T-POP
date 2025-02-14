import numpy as np
import matplotlib.pyplot as plt

# Définition des constantes 
lambda_0 = 632.8e-9  # Longueur d'onde du laser en mètres (632.8 nm)
w_laser = 3.15*(10**-10)  # Taille du faisceau à la sortie du laser en mètres (315 µm)

# Caractéristiques de la fibre
Na = 0.12  # Ouverture numérique
lambda_c = 632*(10**-9)  # Longueur d'onde de coupure en mètres (620 nm)
V_c = 2.405  # Valeur critique de V pour une fibre monomode
t_las = 1.3 * 10**-3 / 2

# Calcul du rayon du coeur de la fibre
a = (V_c * lambda_c) / (2 * np.pi * Na)

# Calcul de la taille du faisceau après la lentille 
def w_image(z):
    return w_laser*np.sqrt(1+(z*t_las/w_laser)**2) #w_1

# Calcul de la taille du mode fondamental de la fibre 
w_0=a * (0.65 + 1.619*(V_c ** -1.5) + 2.879*(V_c ** -6)) #w_2

# Définition de la plage de distances Z 
Z_space = np.linspace(0, 0.01, 1000)  # Distance de 0 à 50 cm 

# Calcul de l'efficacité de couplage T 
T_eff = ((2 * w_image(Z_space) * w_0)/ (w_image(Z_space)** 2 + w_0 ** 2)) ** 2

# Trouver la valeur de Z qui maximise T
Z_optimal_index = np.argmax(T_eff)  # Index du maximum
Z_optimal = Z_space[Z_optimal_index]  # Valeur de Z optimale


# # Trouver la frequence normalisee 
# V_norm =((2 * np.pi * a)/ lambda_0) * Na
# print(V_norm)

# Affichage de la valeur optimale de Z
print(f"Valeur optimale de Z pour maximiser l'efficacité de couplage : {Z_optimal*1000:.3f} mm")

# Tracé de la courbe T(Z) avec la valeur exacte de w_laser
plt.figure(figsize=(8, 6))
plt.plot(Z_space*1000, T_eff, label="Efficacité d'injection T", color="red")
plt.axvline(Z_optimal*1000, color='black', linestyle="--", label=f"Z optimal = {Z_optimal*1000:.3f} mm")
plt.xlabel("Distance Z (mm)")
plt.ylabel("Efficacité de couplage T")
plt.title("Efficacité de couplage en fonction de la distance Z")
plt.grid(True)
plt.legend()
plt.show()



