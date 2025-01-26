import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.optimize import curve_fit

# TODO AJOUT INCERTITUDE

def main():
    #  Lecture du fichier CSV
    path = os.path.dirname(os.path.abspath(__file__))+"\\Images_Lab_Diffraction\\Fente_0_08.csv"
    df = pd.read_csv(path)

    pix_par_mm = 10.6
    mm_par_pix = 1/pix_par_mm
    
    Position = df['pix']
    Intensity = df['Val']

    # Normalisation de l'intensité
    I_max = Intensity.max()
    Intensity_norm = Intensity / I_max

    # Convolution
    Intensity_norm_Convo = Intensity_norm
    window_size = 20
    filter_kernel = np.ones(window_size) / window_size
    Intensity_norm_Convo = convolve(Intensity_norm_Convo, filter_kernel, mode='same')
    window_size = 5
    filter_kernel = np.ones(window_size) / window_size
    Intensity_norm_Convo = convolve(Intensity_norm_Convo, filter_kernel, mode='same')

    # Centrer le data sur le premier max
    convo_max_index = np.argmax(Intensity_norm_Convo)
    Position = Position-convo_max_index

    # Transfo Pix->Position
    Position_mm = Position * mm_par_pix

    # Transfo Position->Angle
    distance_fente_ecran = 0.9
    longueur_donde =  650*(10**(-9))
    angles = np.arctan(Position_mm/1000/distance_fente_ecran)

    # Trouver les Min
    pos_min_pix, _ = find_peaks(1/Intensity_norm_Convo)
    pos_min_mm = ((pos_min_pix-convo_max_index)* mm_par_pix)
    mins_angle = np.arctan(pos_min_mm/1000/distance_fente_ecran)

    # Creation de la list des index des Min
    n=len(pos_min_mm) # Nombre de minimums
    if n%2 == 1: # Le nombre de min doit être pair
        raise ValueError("Minimum number must be even")
    Index_Min = np.linspace(-n/2, n/2, n+1)
    Index_Min = Index_Min[Index_Min != 0]  # Remove the zero

    # Erreur sur les position des min
    pos_err_mm = np.full(len(pos_min_mm), 0.5)#plus petite unite de la regle est le mm
    pos_err_angle = np.arctan(pos_err_mm/1000/distance_fente_ecran)


    # Fit d'une droite sur le data
    def Droite(Mn, pente, offset):
        return Mn*pente + offset
    bounds = ([0, -5], [10, 5])
    initial_guess = [5, 0]
    params, covariance = curve_fit(Droite, Index_Min, mins_angle, p0=initial_guess, bounds=bounds)

    Fitted_line = Droite(Index_Min, params[0], params[1])

    a = longueur_donde/params[0]
    print(f'Variance = {covariance}')
    print(f'Fente de {a*1000}mm')

    # Graphique du patron
    plt.subplot(1, 3, 1)  # (rows, cols, index)

    # En Angle
    plt.plot(angles, Intensity_norm, label="Intensité normalisée", color="red")
    # plt.plot(angles, Intensity_norm_Convo, label="Convolution Intensité normalisée", color="blue")

    # Mise en forme du graphique
    plt.xlabel("Positon (rad)")
    plt.ylabel("I/I₀")
    plt.title("Intensité relative du patron en fonction de la position angulaire")
    plt.legend()
    plt.grid(True)

    # Graph des Min
    plt.subplot(1, 3, (2,3))
    plt.scatter(Index_Min, mins_angle, label="Minimums", color="red", marker='x') # En Angle
    # plt.errorbar(Index_Min, mins_angle, yerr=pos_err_angle, fmt='o', label="Minimums", color="blue", ecolor="blue", capsize=10, markersize=1)
    plt.plot(Index_Min, Fitted_line, label="Fit Linéaire", color="blue")
    plt.plot(Index_Min, Index_Min*longueur_donde/(0.08*(10**-3)), label="Valeur Théorique", color="green")
    plt.xlabel("mₙ")
    plt.xticks(Index_Min)
    plt.ylabel("Position (rad)")
    plt.title("Position angulaire des minimums du patron en fonction de leur indice mₙ")

    # Determine the range of data
    x_min, x_max = Index_Min.min(), Index_Min.max()
    y_min, y_max = mins_angle.min(), mins_angle.max()
    # Add the equation to the graph
    round_at = 5
    equation = f"FIT: y = {round(params[0],round_at)}x + {round(params[1], round_at)}"
    plt.text(
        x_min + (x_max - x_min) * 0.22,  # x: 70% from the left
        y_min + (y_max - y_min) * 0.78,  # y: 10% above the bottom
        equation,
        fontsize=12,
        color="blue"
    )

    plt.legend()
    plt.grid(True)

    plt.suptitle("Analyse du patron produit par un laser 650nm passant dans une fente de 0.08mm", fontsize=16)

    # Affichage
    plt.subplots_adjust(
        top=0.9,
        bottom=0.09,
        left=0.045,
        right=0.98,
        hspace=0.0,
        wspace=0.2
    )
    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
