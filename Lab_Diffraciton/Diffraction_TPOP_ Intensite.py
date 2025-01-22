import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from scipy.signal import find_peaks
import os

# TODO pour les min et les max: mettre la position sur l'axe des y et l'index des max-min sur l'axe des y: la pente de ce graph est lambda/a

def main():
    #  Lecture du fichier CSV
    path = os.path.dirname(os.path.abspath(__file__))+"\\Images_Lab_Diffraction\\Fente_0_08.csv"
    # df = pd.read_csv("C:/Users/HP/Documents/ULaval_S4/Travaux_Opt/Lab_Diffraciton/Images_Lab_Diffraction/Fente_0_08.csv")  # sep=';' si besoin
    df = pd.read_csv(path)
    # print(os.path.dirname(os.path.abspath(__file__))+"Images_Lab_Diffraction/Fente_0_08.csv")

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
    Position_mm = Position * 52 / 684

    list_min, _ = find_peaks(1/Intensity_norm_Convo)
    list_max, _ = find_peaks(Intensity_norm_Convo)

    # Tracé des courbes
    plt.figure(figsize=(8, 5))

    plt.plot(Position_mm, Intensity_norm, label="Intensité normalisée", color="red")
    plt.plot(Position_mm, Intensity_norm_Convo, label="Convolution Intensité normalisée", color="blue")

    # Display min
    plt.axvline(x=((list_min-convo_max_index)* 52 / 684)[0], color='green', linestyle='-', label="Min")
    for min in ((list_min-convo_max_index)* 52 / 684)[1:]:
        plt.axvline(x=min, color='green', linestyle='-')

    # Display max
    plt.axvline(x=((list_max-convo_max_index)* 52 / 684)[0], color='purple', linestyle='-', label="Max")
    for max in ((list_max-convo_max_index)* 52 / 684)[1:]:
        plt.axvline(x=max, color='purple', linestyle='-')

    # Mise en forme du graphique
    # plt.xlabel("Position (pixels)")
    plt.xlabel("Position (mm)")
    plt.ylabel("I/I₀")
    plt.title("Figure de diffraction - Intensité en fonction de la position")
    plt.legend()
    plt.grid(True)

    # Affichage
    plt.show()

if __name__ == "__main__":
    main()
