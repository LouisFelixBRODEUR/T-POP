import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import EngFormatter
from CoolFormat import NoSpaceEngFormatter


data_path = os.path.dirname(os.path.abspath(__file__))+"\\lab_data.csv"

# A=TM & B=TE

# Extract data from csv
df = pd.read_csv(data_path, skiprows=1, header=None)
angle_A = df[1][1:].values.astype(float)
data_A = df[2][1:].values.astype(float)
angle_B = df[4][1:].values.astype(float)
data_B = df[5][1:].values.astype(float)

# Fit Theorical Models

# Plot Data
plt.figure(figsize=(8, 6))
plt.plot(angle_A, data_A, marker='o', linestyle='', label="Intensité TM")
plt.plot(angle_B, data_B, marker='x', linestyle='', label="Intensité TE")

#X Axis
plt.xlabel("Angle")
plt.gca().xaxis.set_major_formatter(NoSpaceEngFormatter(unit='°'))
plt.xlim(0, 90)

# Y Axis
plt.ylabel("Intensité")
plt.gca().yaxis.set_major_formatter(NoSpaceEngFormatter(unit='µA'))

# plt.yscale('log')
# plt.ylim(None, 1.5)

# Title
plt.title("Intensité mesurée en fonction de l'angle d'incidence pour \n les composantes perpendiculaire(TE) et parallèle(TM) au plan d'incidence")
plt.grid(True)
plt.legend()

# Show the plot
plt.show()



