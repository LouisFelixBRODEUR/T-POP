import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import EngFormatter
from CoolFormat import NoSpaceEngFormatter
from scipy.optimize import curve_fit

'''
FYI:
    A = TM = pol para
    B = TE = pol perp
'''

data_path = os.path.dirname(os.path.abspath(__file__))+"\\lab_data.csv"

# Extract data from csv
df = pd.read_csv(data_path, skiprows=1, header=None)
angle_A = df[1][1:].values.astype(float)
angle_A = angle_A[~np.isnan(angle_A)]
data_A = df[2][1:].values.astype(float)
data_A = data_A[~np.isnan(data_A)]
angle_B = df[4][1:].values.astype(float)
angle_B = angle_B[~np.isnan(angle_B)]
data_B = df[5][1:].values.astype(float)
data_B = data_B[~np.isnan(data_B)]

# Relative Intensite
data_A = data_A/np.max(data_A)
data_B = data_B/np.max(data_B)

# Fit Theorical Models
def Para_Reflectivite(theta1, n2):
    n1 = 1
    theta1 = theta1/360*2*np.pi
    theta2 = np.arcsin(np.sin(theta1)/n2)
    r_para = (n1*np.cos(theta2)-n2*np.cos(theta1))/(n1*np.cos(theta2)+n2*np.cos(theta1))
    R_para = r_para**2
    return R_para
bounds = ([1], [5])
initial_guess = [1.5]
params, covariance = curve_fit(Para_Reflectivite, angle_A, data_A, p0=initial_guess, bounds=bounds)
n2_A = params[0]
n2_A_uncertainty = np.sqrt(covariance[0, 0])
A_data_Fitted = Para_Reflectivite(angle_A, n2_A)

def Perp_Reflectivite(theta1, n2):
    n1 = 1
    theta1 = theta1/360*2*np.pi
    theta2 = np.arcsin(np.sin(theta1)/n2)
    r_perp = (n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2))
    R_perp = r_perp**2
    return R_perp
bounds = ([1], [5])
initial_guess = [1.5]
params, covariance = curve_fit(Perp_Reflectivite, angle_B, data_B, p0=initial_guess, bounds=bounds)
n2_B = params[0]
n2_B_uncertainty = np.sqrt(covariance[0, 0])
B_data_Fitted = Perp_Reflectivite(angle_B, n2_B)


# Compute Brewster avec Incertitudes
def Compute_Brewster(n2, n2_err):
    Value  = np.arctan(n2)/np.pi/2*360
    Max_Value = np.arctan(n2+n2_err)/np.pi/2*360
    Min_Value = np.arctan(n2-n2_err)/np.pi/2*360
    return Value, Max_Value-Value, Value-Min_Value
a,b,c = Compute_Brewster(n2_A, n2_A_uncertainty)
print(f'Brewster TM : {a:.4f} ± {(b+c)/2:.4f}')
a,b,c = Compute_Brewster(n2_B, n2_B_uncertainty)
print(f'Brewster TE : {a:.4f} ± {(b+c)/2:.4f}')

# Plot Data
plt.figure(figsize=(8, 6))
plt.plot(angle_A, data_A, marker='o', linestyle='', label="Intensité TM", color = 'blue', markersize=8)
plt.plot(angle_A, A_data_Fitted, marker='', linestyle=':', label=f"Intensité TM Fit (n₂={n2_A:.4f} ± {n2_A_uncertainty:.4f})", color = 'blue', linewidth=2)
plt.plot(angle_B, data_B, marker='^', linestyle='', label="Intensité TE", color = 'orange', markersize=8)
plt.plot(angle_B, B_data_Fitted, marker='', linestyle='--', label=f"Intensité TE Fit (n₂={n2_B:.4f} ± {n2_B_uncertainty:.4f})", color = 'orange', linewidth=2)

#X Axis
plt.xlabel("Angle", fontsize=25)
plt.gca().xaxis.set_major_formatter(NoSpaceEngFormatter(unit='°'))
plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
plt.xlim(0, 95)

# Y Axis
plt.ylabel("Intensité Relative", fontsize=25)
# plt.gca().yaxis.set_major_formatter(NoSpaceEngFormatter(unit='µA'))
# plt.yscale('log')
# plt.ylim(None, 1.5)

# General
# plt.title("Intensité mesurée en fonction de l'angle d'incidence pour \n les composantes perpendiculaire(TE) et parallèle(TM) au plan d'incidence")
plt.grid(True)
plt.legend(fontsize=18)
plt.show()



