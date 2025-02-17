# pip install numpy matplotlib pillow scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.optimize import curve_fit
from scipy.special import jn, jn_zeros


# List of files to process
Hautement_Multimode_path = os.path.dirname(os.path.abspath(__file__))+"\\Lumi_Hautement_Multimode_Crop.tif",
Legerement_Multimode_path = os.path.dirname(os.path.abspath(__file__))+"\\Lumi_Legerement_Multimode_Crop.tif",
Monomode_path = os.path.dirname(os.path.abspath(__file__))+"\\Lumi_Monomode_Crop.tif"

file = Monomode_path

img = Image.open(file)
data = np.array(img)

# Find 3D center
x_range=(120, 210)
y_range=(120, 210)
sliced_data = data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
max_row_positions = np.argmax(sliced_data, axis=1)
max_col_positions = np.argmax(sliced_data, axis=0)
avg_x = np.mean(max_row_positions) + x_range[0]
avg_y = np.mean(max_col_positions) + y_range[0]
center =  avg_x, avg_y

# Center the data
center_row, center_col = center[1],center[0]
num_rows, num_cols = data.shape
max_row_distance = min(center_row, num_rows - 1 - center_row)
max_col_distance = min(center_col, num_cols - 1 - center_col)
submatrix_size_rows = 2 * max_row_distance + 1
submatrix_size_cols = 2 * max_col_distance + 1
start_row = int(center_row - max_row_distance)
end_row = int(center_row + max_row_distance + 1)
start_col = int(center_col - max_col_distance)
end_col = int(center_col + max_col_distance + 1)
centered_data = data[start_row:end_row, start_col:end_col]

title = file.split('\\')[-1]
# # HeatMap
# plt.figure(figsize=(8, 6))
# plt.imshow(data, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.scatter(center[0], center[1], color='green', marker='X', s=100, label='Center')
# plt.title(f'Heatmap - {title} NOT CENTERED')
# # HeatMap
# plt.figure(figsize=(8, 6))
# plt.imshow(centered_data, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.scatter(len(centered_data[0])//2, len(centered_data)//2, color='green', marker='X', s=100, label='Center')
# plt.title(f'Heatmap - {title}')
# plt.show()

# profil radial de la gaussian
gaussian_image = centered_data
size_x, size_y = len(centered_data[0]), len(centered_data)
center_x, center_y = size_x//2, size_y//2
x = np.arange(size_x) - center_x
y = np.arange(size_y) - center_y
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2 + yy**2)  # Radial distance from the center
theta = np.arctan2(yy, xx)  # Angle in radians (-pi to pi)
r_flat = r.flatten()
theta_flat = theta.flatten()
intensity_flat = gaussian_image.flatten()
r_max = np.max(r)
r_bins = np.linspace(0, r_max, 100)  # Adjust the number of bins as needed
r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2  # Center of each bin
bin_indices = np.digitize(r_flat, r_bins)
radial_profile = np.array([intensity_flat[bin_indices == i].mean() for i in range(1, len(r_bins))])
r_combined = np.concatenate([-r_bin_centers[::-1], r_bin_centers])
intensity_combined = np.concatenate([radial_profile[::-1], radial_profile])  # Reverse intensity

# Cut Edges
r_min = -210
r_max = 210
mask = (r_combined >= r_min) & (r_combined <= r_max)
# Apply the mask to filter r_combined and intensity_combined
r_combined = r_combined[mask]
intensity_combined = intensity_combined[mask]

# drop_data_to make tendre vers 0
intensity_combined = intensity_combined-22.8

# Intensité relative
intensity_combined = intensity_combined/np.max(intensity_combined)

# Fit des 2 premiers modes: LP01 & LP11
def intensity_profile(r, w_0, I_1, beta_1):
    first_zero = jn_zeros(1, 1)[0] / beta_1  # First zero of J_1 divided by beta_1
    Intensite_LP01 = np.exp(-2 * r**2 / w_0**2)
    Intensite_LP11 = I_1 * (jn(1, beta_1 * r))**2
    total_intensity = Intensite_LP01 + Intensite_LP11
    total_intensity[np.abs(r) > first_zero] = 0
    return total_intensity

bounds = ([0,0,0], [300,1,0.1])
initial_guess = [50,0.5,0.01]
params, covariance = curve_fit(intensity_profile, r_combined, intensity_combined, p0=initial_guess, bounds=bounds)
print(params)
w_0, Int_1, beta_1 = params
Fit_Intensity = intensity_profile(r_combined, w_0, Int_1, beta_1)

# Affichage du résultat
plt.figure(figsize=(10, 5))
plt.plot(r_combined, intensity_combined, label="Profil Radial", linestyle="-", c='blue')
plt.plot(r_combined, Fit_Intensity, label="Profil Radial Fit (LP01 & LP11)", linestyle="--", c='red')
plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel("Distance radiale (pixels)", fontsize=25)
plt.ylabel("Intensité normalisée (I/Iₘₐₓ)", fontsize=25)
plt.legend(fontsize=18)
plt.grid()
plt.show()