import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from scipy.signal import find_peaks
import os
from scipy.optimize import curve_fit


def main():
    #  Lecture du fichier CSV
    path = os.path.dirname(os.path.abspath(__file__))+"\\Images_Lab_Diffraction\\Fente_0_08.csv"
    df = pd.read_csv(path)

    
    # # Graphique du patron
    # plt.subplot(1, 3, 1)  # (rows, cols, index)

    # # En Angle
    # plt.plot(angles, Intensity_norm, label="Intensité normalisée", color="red")
    # # plt.plot(angles, Intensity_norm_Convo, label="Convolution Intensité normalisée", color="blue")

    # # Mise en forme du graphique
    # plt.xlabel("Positon (rad)")
    # plt.ylabel("I/I₀")
    # plt.title("Intensité relative du patron en fonction de la position angulaire")
    # plt.legend()
    # plt.grid(True)

    # # Graph des Min
    # plt.subplot(1, 3, (2,3))
    # plt.scatter(Index_Min, mins_angle, label="Minimums", color="red", marker='x') # En Angle
    # # plt.errorbar(Index_Min, mins_angle, yerr=pos_err_angle, fmt='o', label="Minimums", color="blue", ecolor="blue", capsize=10, markersize=1)
    # plt.plot(Index_Min, Fitted_line, label="Fit Linéaire", color="blue")
    # plt.plot(Index_Min, Index_Min*longueur_donde/(0.08*(10**-3)), label="Valeur Théorique", color="green")
    # plt.xlabel("mₙ")
    # plt.xticks(Index_Min)
    # plt.ylabel("Position (rad)")
    # plt.title("Position angulaire des minimums du patron en fonction de leur indice mₙ")

    # # Determine the range of data
    # x_min, x_max = Index_Min.min(), Index_Min.max()
    # y_min, y_max = mins_angle.min(), mins_angle.max()
    # # Add the equation to the graph
    # round_at = 5
    # equation = f"FIT: y = {round(params[0],round_at)}x + {round(params[1], round_at)}"
    # plt.text(
    #     x_min + (x_max - x_min) * 0.22,  # x: 70% from the left
    #     y_min + (y_max - y_min) * 0.78,  # y: 10% above the bottom
    #     equation,
    #     fontsize=12,
    #     color="blue"
    # )

    # plt.legend()
    # plt.grid(True)

    # plt.suptitle("Analyse du patron produit par un laser 650nm passant dans une fente de 0.08mm", fontsize=16)

    # # Affichage
    # plt.subplots_adjust(
    #     top=0.9,
    #     bottom=0.09,
    #     left=0.045,
    #     right=0.98,
    #     hspace=0.0,
    #     wspace=0.2
    # )
    # # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
