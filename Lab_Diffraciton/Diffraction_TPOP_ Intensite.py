import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.optimize import curve_fit

# TODO Formater les graph pour la presentation
# TODO AJOUT INCERTITUDE
# TODO AJOUT EQUATION DU FIT
# TODO AJOUT DES CALCULS DE LA TAILLE DE LA FENTE????

def main():
    #  Lecture du fichier CSV
    path = os.path.dirname(os.path.abspath(__file__))+"\\Images_Lab_Diffraction\\Fente_0_08.csv"
    df = pd.read_csv(path)

    # mm_par_pix = 52 / 684 #REAL
    mm_par_pix = 66 / 684 #FAKE

    Position = df['pix']
    Intensity = df['Val']

    # Normalisation de l'intensité
    I_max = Intensity.max()
    Intensity_norm = Intensity / I_max

    # Convolution
    window_size = 20
    filter_kernel = np.ones(window_size) / window_size
    Intensity_norm_Convo = convolve(Intensity_norm, filter_kernel, mode='same')
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
    mins_val = [Intensity_norm[i] for i in pos_min_pix]

    # Creation de la list des index des Min
    n=len(pos_min_mm) # Nombre de minimums
    if n%2 == 1: # Le nombre de min doit être pair
        raise ValueError("Minimum number must be even")
    Index = np.linspace(-n/2, n/2, n+1)
    Index = Index[Index != 0]  # Remove the zero

    # Fit d'une droite sur le data
    def Droite(Mn, pente, offset):
        return Mn*pente + offset
    bounds = ([0, -5], [10, 5])
    initial_guess = [5, 0]
    params, covariance = curve_fit(Droite, Index, mins_angle, p0=initial_guess, bounds=bounds)
    print(params)

    Fitted_line = Droite(Index, params[0], params[1])

    a = longueur_donde/params[0]
    print(f'Fente de {a*1000}mm')


    # Graphique du patron
    plt.subplot(1, 2, 1)  # (rows, cols, index)

    # En Angle
    plt.plot(angles, Intensity_norm, label="Intensité normalisée", color="red")
    # plt.plot(angles, Intensity_norm_Convo, label="Convolution Intensité normalisée", color="blue")

    # En Position
    # plt.plot(Position_mm, Intensity_norm, label="Intensité normalisée", color="red")
    # plt.plot(Position_mm, Intensity_norm_Convo, label="Convolution Intensité normalisée", color="blue")

    # Display Min points
    # plt.scatter(pos_min_mm, mins_val, color='green', marker='x', label='Minimums')

    # Mise en forme du graphique
    # plt.xlabel("Position (mm)")
    plt.xlabel("Angle (rad)")
    plt.ylabel("I/I₀")
    plt.title("Intensité relative en fonction de la position angulaire du patron produit par\n un laser 650nm passant dans une fente de 0.08mm")
    plt.legend()
    plt.grid(True)

    # Graph des Min
    plt.subplot(1, 2, 2)
    # plt.scatter(Index, pos_min_mm, label="Min", color="blue") # En Position
    plt.scatter(Index, mins_angle, label="Min", color="blue") # En Angle
    plt.plot(Index, Fitted_line, label="Fit Linéaire", color="green")
    plt.xlabel("mₙ")
    # plt.ylabel("Position (mm)")
    plt.ylabel("Angle (rad)")
    plt.title("Position angulaire des minimums du pattron en fonction de leur indice mₙ")
    plt.legend()
    plt.grid(True)

    # Affichage
    # plt.subplots_adjust(
    #     top=0.9,      # Adjust the top spacing
    #     bottom=0.08,  # Adjust the bottom spacing
    #     left=0.07,    # Adjust the left spacing
    #     right=0.975,  # Adjust the right spacing
    #     hspace=0.0,   # No vertical space between subplots
    #     wspace=0.305  # Horizontal space between subplots
    # )
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
