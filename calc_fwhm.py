import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data
df = pd.read_excel("outfiles_pos_calc_0103_pixdim3/0_2_pos.xlsx")
x_values = df["X"].values
y_values = df["Y"].values

# Define a 1D Gaussian function
def gaussian(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# Compute histogram of x-values
hist_x, bin_edges_x = np.histogram(x_values, bins=50, density=True)
bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2

# Compute histogram of y-values
hist_y, bin_edges_y = np.histogram(y_values, bins=50, density=True)
bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

# Initial guesses for Gaussian fit
p0_x = [max(hist_x), np.mean(x_values), np.std(x_values)]
p0_y = [max(hist_y), np.mean(y_values), np.std(y_values)]

# Fit Gaussian to histograms
popt_x, pcov_x = curve_fit(gaussian, bin_centers_x, hist_x, p0=p0_x)
popt_y, pcov_y = curve_fit(gaussian, bin_centers_y, hist_y, p0=p0_y)

# Extract standard deviations and compute FWHM
sigma_x = np.abs(popt_x[2])
sigma_y = np.abs(popt_y[2])

fwhm_x = 2.355 * sigma_x
fwhm_y = 2.355 * sigma_y

# Compute errors from covariance matrix
sigma_x_err = np.sqrt(np.abs(pcov_x[2, 2]))
sigma_y_err = np.sqrt(np.abs(pcov_y[2, 2]))

fwhm_x_err = 2.355 * sigma_x_err
fwhm_y_err = 2.355 * sigma_y_err

# Print results
print(f"FWHM_x: {fwhm_x:.3f} ± {fwhm_x_err:.3f}")
print(f"FWHM_y: {fwhm_y:.3f} ± {fwhm_y_err:.3f}")
print(f"FWHM_X,ux,FWHM_Y,UY: {fwhm_x:.3f},{fwhm_x_err:.3f},{fwhm_y:.3f},{fwhm_y_err:.3f}")

# Generate smooth curve for plotting
x_fit = np.linspace(min(x_values), max(x_values), 500)
y_fit = np.linspace(min(y_values), max(y_values), 500)

gauss_x = gaussian(x_fit, *popt_x)
gauss_y = gaussian(y_fit, *popt_y)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot histogram and Gaussian fit for X
axes[0].bar(bin_centers_x, hist_x, width=(bin_edges_x[1] - bin_edges_x[0]), alpha=0.6, color='blue', label="Binned Data")
axes[0].plot(x_fit, gauss_x, 'r-', label="Fitted Gaussian", linewidth=2)
axes[0].set_title("Gaussian Fit for X Values")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Density")
axes[0].legend()

# Plot histogram and Gaussian fit for Y
axes[1].bar(bin_centers_y, hist_y, width=(bin_edges_y[1] - bin_edges_y[0]), alpha=0.6, color='green', label="Binned Data")
axes[1].plot(y_fit, gauss_y, 'r-', label="Fitted Gaussian", linewidth=2)
axes[1].set_title("Gaussian Fit for Y Values")
axes[1].set_xlabel("Y")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.show()
