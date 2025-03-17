import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class PlotterClass:
    def __init__(self, 
                 file_path: str, 
                 axis: Optional[str], 
                 base_folder_to_save: Optional[str]):
        self.file_path = file_path
        self.axis = axis
        self.base_folder_to_save = base_folder_to_save or os.getenv("base_folder")
        
        if self.axis == "x":
            self.independent_var = "y"
        elif self.axis == "y":
            self.independent_var = "x"

    def _check_file_exists(self, file_path):
        """Helper method to check if a file exists."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    def _ensure_directory_exists(self, save_path):
        """Helper method to ensure directories exist before saving."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def fwhm(self):
        self._check_file_exists(self.file_path)
        df = pd.read_csv(self.file_path)

        axis = self.axis.lower()
        independent_var = self.independent_var.lower()

        df = df[df[independent_var] == 0]

        # Extract relevant columns
        values = df[axis].values
        fwhm_values = df[f"fwhm_{axis}"].values
        uncertainties = df[f"u_fwhm_{axis}"].values

        # Plot
        plt.figure(figsize=(8, 5))
        plt.errorbar(values, fwhm_values, yerr=uncertainties, fmt='o', capsize=5, capthick=1.5, label=f"FWHM_{axis.upper()} ± Uncertainty")
        plt.xlabel(axis.upper())
        plt.ylabel(f"FWHM_{axis.upper()}")
        plt.title(f"FWHM_{axis.upper()} as a function of {axis.upper()} with Uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the plot
        save_path = os.path.join(self.base_folder_to_save, "images", f"fwhm_{axis}.png")
        self._ensure_directory_exists(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def diff(self, is_heatmap_plot):
        self._check_file_exists(self.path)
        df = pd.read_csv(self.path, usecols=['X_exp', 'Y_exp', 'X_diff', 'Y_diff'])
        df = df[df[f"{self.independent_var.upper()}_exp"] == 0]

        axis = self.axis.lower()
        if is_heatmap_plot:
            plt.figure(figsize=(10, 6))
            plt.title(f"Position deviation in the {axis.upper()} axis with {self.independent_var.upper()}=0")
            plt.xlabel(f"{axis.upper()} expected (mm)")
            plt.ylabel(f"{axis.upper()} expected - {axis.upper()} calculated (mm)")
            plt.hist2d(df[f"{axis.upper()}_exp"], df[f"{axis.upper()}_diff"], bins=40, cmap='YlOrRd')
            plt.ylim(-4, 4)
            plt.colorbar(label='Frequency')
            plt.grid(True)
            plt.show()

            save_path = os.path.join(self.base_folder_to_save, "images", f"heatmap_diff_{axis}.png")
            self._ensure_directory_exists(save_path)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            grouped = df.groupby(f'{axis.upper()}_exp')[f'{axis.upper()}_diff']
            average_diff = grouped.mean()
            std_diff = grouped.std()

            plt.figure(figsize=(10, 6))
            plt.title(f'Average {axis.upper()}_diff vs {axis.upper()}_exp ({self.independent_var.upper()}_exp = 0) with Error Bars (std)')
            plt.xlabel(f'{axis.upper()} Expected (mm)')
            plt.ylabel(f'Average {axis.upper()}_diff (mm)')
            plt.errorbar(average_diff.index, average_diff.values, yerr=std_diff.values, fmt='o', color='blue', capsize=5, label=f'Average {axis.upper()}_diff')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            save_path = os.path.join(self.base_folder_to_save, "images", f"diff_{axis}.png")
            self._ensure_directory_exists(save_path)
            plt.savefig(save_path, dpi=300)
            plt.close()

    def rows_columns_yield(self, row_column, x_value, y_value):
        file_name = f"{x_value}_{y_value}.csv"
        file_path = os.path.join(self.base_folder_to_save, file_name)
        self._check_file_exists(file_path)

        df = pd.read_csv(file_path, usecols=[row_column])
        filtered_data = df[(df[row_column] > 0) & (df[row_column] < 3000)]

        plt.figure(figsize=(8, 6))
        plt.title(f"{row_column} Yield")
        plt.xlabel('Yield')
        plt.ylabel('Number of events')
        plt.hist(filtered_data[row_column], bins=50)
        plt.grid(True)
        plt.show()

        save_path = os.path.join(self.base_folder_to_save, "images", f"rows_columns_yield_{x_value}_{y_value}.png")
        self._ensure_directory_exists(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def yx_pos_distribution(self, x_value, y_value):
        self._check_file_exists(self.path)
        df = pd.read_csv(self.path, usecols=['X', 'Y', 'X_exp', 'Y_exp'])

        axis = self.axis.lower()
        filtered_data = df[(df['X_exp'] == x_value) & (df['Y_exp'] == y_value)]

        plt.figure(figsize=(8, 6))
        plt.title(f'Distribution of estimated positions in the {axis.upper()}-axis for the data point ({x_value},{y_value})')
        plt.xlabel(f'{axis.upper()} (mm)')
        plt.ylabel('Number of events')
        plt.hist(filtered_data[axis], bins=50)
        plt.grid(True)
        plt.show()

    def YX_heatmap(self, is_single_plot, x_value=0, y_value=0):
        file_name = f"{x_value}_{y_value}_pos.xlsx"
        file_path = os.path.join(self.base_folder_to_save, file_name)
        self._check_file_exists(file_path)

        df = pd.read_excel(file_path, usecols=['X', 'Y'])

        if is_single_plot:
            plt.figure(figsize=(8, 6))
            plt.title(f'({x_value},{y_value}) arrange')
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.hist2d(df['X'], df['Y'], bins=20, cmap='YlOrRd')
            plt.colorbar(label='Frequency')
            plt.grid(True)
            plt.show()
        else:
            xtab = np.array([-6, -4, -2, 0, 2, 4, 6])
            ytab = np.array([-6, -4, -2, 0, 2, 4])

            for x in xtab:
                for y in ytab:
                    file_name = f"{x}_{y}_pos.xlsx"
                    file_path = os.path.join(self.base_folder_to_save, file_name)
                    self._check_file_exists(file_path)

                    df = pd.read_excel(file_path, usecols=['X', 'Y'])

                    plt.figure(figsize=(8, 6))
                    plt.title(f'({x},{y}) arrange')
                    plt.xlabel('X (mm)')
                    plt.ylabel('Y (mm)')
                    plt.hist2d(df['X'], df['Y'], bins=20, cmap='YlOrRd')
                    plt.colorbar(label='Frequency')
                    plt.grid(True)

                    save_path = os.path.join(self.base_folder_to_save, "position_images", f"{x}_{y}_distribution.png")
                    self._ensure_directory_exists(save_path)
                    plt.savefig(save_path, dpi=300)
                    plt.close()
