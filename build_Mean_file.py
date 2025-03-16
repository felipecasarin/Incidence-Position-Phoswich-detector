import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Define Gaussian function
def gaussian(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# Store results
gaussian_results = {}
mean_df=[]
umean_df=[]

xtab=np.array([-6, -4, -2, 0, 2, 4, 6])
ytab=np.array([-6, -4, -2, 0, 2, 4])

for x in xtab:
    for y in ytab:
        # Load data
        df = pd.read_csv(f"collection_files_csv_unfiltered/{x}_{y}.csv")

        # Create a figure for subplots
        fig, axes = plt.subplots(nrows=len(df.columns), figsize=(8, 5 * len(df.columns)))

        if len(df.columns) == 1:  # Ensure `axes` is iterable even for one column
            axes = [axes]

        for ax, col in zip(axes, df.columns):
            # Filter data
            filtered_data = df[(df[col] > 0) & (df[col] < 3000)][col]

            # Histogram
            hist, bin_edges_x = np.histogram(filtered_data, bins=50, density=True)
            bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2

            # Initial parameter guess
            p0_x = [max(hist), np.mean(filtered_data), np.std(filtered_data)]

            try:
                popt, pcov = curve_fit(gaussian, bin_centers_x, hist, p0=p0_x)
                mean, mean_err = popt[1], np.sqrt(pcov[1, 1]) if pcov is not None else np.nan
                gaussian_results[col] = (mean, mean_err)
                mean_df.append(mean)
                umean_df.append(mean_err)

                # Plot histogram
                ax.hist(filtered_data, bins=50, density=True, alpha=0.6, color='b', label="Data")

                # Plot fitted Gaussian
                x_fit = np.linspace(min(bin_centers_x), max(bin_centers_x), 200)
                ax.plot(x_fit, gaussian(x_fit, *popt), 'r-', label=f'Gaussian Fit\nμ={mean:.2f} ± {mean_err:.2f}')

            except RuntimeError:
                print(f"Curve fitting failed for column: {col}")
                gaussian_results[col] = (np.nan, np.nan)

            # Label plot
            ax.set_title(f"Histogram & Gaussian Fit for {col}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()

        # Show the plots
        save_path = os.path.join(f"build_mean_file/", f"{x}_{y}_L_C_distribution_gauss_fit.png")
        plt.savefig(save_path, dpi=300)
        plt.close() 

# Save Mean and Error to txt files
with open("build_mean_file/09_03_Mean.txt", "w") as f:
    for i in range(0, len(mean_df), 8):
        f.write(" ".join(map(str, mean_df[i:i+8])) + "\n")
        

with open("build_mean_file/09_03_Error.txt", "w") as f:
    for i in range(0, len(umean_df), 8):
        f.write(" ".join(map(str, umean_df[i:i+8])) + "\n")

# Print results
for col, (mu, mu_err) in gaussian_results.items():
    print(f"{col}: mean = {mu:.4f}, error = {mu_err:.4f}")
