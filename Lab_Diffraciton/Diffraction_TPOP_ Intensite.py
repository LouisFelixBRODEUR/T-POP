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

    # Trouver les Min et Max
    list_min, _ = find_peaks(1/Intensity_norm_Convo)
    list_min_replace = ((list_min-convo_max_index)* 52 / 684)
    list_max, _ = find_peaks(Intensity_norm_Convo)
    list_max_replace = ((list_max-convo_max_index)* 52 / 684)
    mins_val = [Intensity_norm[i] for i in list_min]
    maxs_val = [Intensity_norm[i] for i in list_max]

    # Graphique du patron
    plt.subplot(1, 3, 1)  # (rows, cols, index)
    plt.plot(Position_mm, Intensity_norm, label="Intensité normalisée", color="red")
    plt.plot(Position_mm, Intensity_norm_Convo, label="Convolution Intensité normalisée", color="blue")
    # Display Min Max points
    plt.scatter(list_min_replace, mins_val, color='green', marker='x', label='Minimums')
    plt.scatter(list_max_replace, maxs_val, color='purple', marker='x', label='Maximums')
    # Mise en forme du graphique
    plt.xlabel("Position (mm)")
    plt.ylabel("I/I₀")
    plt.title("Intensité relative en fonction de la distance avec le centre du patron produit par\n un laser 650nm passant dans une fente de 0.08mm")
    plt.legend()
    plt.grid(True)

    # Graphique des Min
    n=len(list_min_replace)
    Index = np.linspace(-((n-1)//2), (n-1)//2, n)  # Create a linspace array
    Index = Index[Index != 0]  # Remove the zero
    plt.subplot(1, 3, 2)
    plt.scatter(Index, list_min_replace, label="Min", color="blue")
    plt.xlabel("mₙ")
    plt.ylabel("Position (mm)")
    plt.title("Position des minimum du pattron en fonction de leur indice")
    plt.legend()
    plt.grid(True)

    # Graphique des Max
    n=len(list_max_replace)
    Index = np.linspace(-n/2, n/2, n, endpoint=True)
    Index = Index[(Index != 0.5) & (Index != -0.5)]
    plt.subplot(1, 3, 3)
    plt.scatter(Index, list_max_replace, label="Max", color="blue")
    plt.xlabel("mₙ")
    plt.ylabel("Position (mm)")
    plt.title("Position des maximums du pattron en fonction de leur indice")
    plt.legend()
    plt.grid(True)

    # Affichage
    plt.subplots_adjust(
        top=0.9,      # Adjust the top spacing
        bottom=0.08,  # Adjust the bottom spacing
        left=0.07,    # Adjust the left spacing
        right=0.975,  # Adjust the right spacing
        hspace=0.0,   # No vertical space between subplots
        wspace=0.305  # Horizontal space between subplots
    )
    plt.show()
    

if __name__ == "__main__":
    main()
