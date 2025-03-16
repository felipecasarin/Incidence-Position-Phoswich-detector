import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os

class PlotterClass:
    def __init__(self, path, axis):
        self.path = path
        self.axis = axis
        if self.axis == "x":
            self.independent_var = "y"
        elif self.axis == "y":
            self.independent_var = "x"
    
    def fwhm(self):
        df = pd.read_csv(self.path)  # "pixdim3_fwhm.csv"

        axis = self.axis.lower()

        independent_var = self.independent_var.lower()
        

        df = df[df[f"{independent_var}"]==0]

        # Extract relevant columns
        values = df[f"{axis}"].values
        fwhm_values = df[f"fwhm_{axis}"].values
        uncertainties = df[f"u_fwhm_{axis}"].values

        # Plot
        plt.figure(figsize=(8, 5))
        plt.errorbar(values, fwhm_values, yerr=uncertainties, fmt='o', capsize=5, capthick=1.5, label=f"FWHM_{axis.upper()} ± Uncertainty")

        # Labels and title
        plt.xlabel(f"{axis.upper()}")
        plt.ylabel(f"FWHM_{axis.upper()}")
        plt.title(f"FWHM_{axis.upper()} as a function of {axis.upper()} with Uncertainty")
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()
    

    def heatmap_diff(self, is_heatmap_plot):

        axis = self.axis.lower()
        independent_var = self.independent_var.lower()
        file_path = self.path   # 'C:/Users/Felipe Casarin/Desktop/Incidence-Position-On-Phoswich-Detector/diff_values_0103_pixdim3.csv'

        df = pd.read_csv(file_path, usecols=['X_exp', 'Y_exp', 'X_diff', 'Y_diff'])
        df = df[df[f"{self.independent_var.upper()}_exp"] == 0]

        if is_heatmap_plot:
            # Used to display the difference between expected and obtained positions in the given direction, with heatmap display (density of events)

            # Plotting the heatmap
            plt.figure(figsize=(10, 6))
            plt.title(f"Position deviation in the {axis.upper()} axis with {independent_var.upper()}=0")
            plt.xlabel(f"{axis.upper()} expected (mm)")
            plt.ylabel(f"{axis.upper()} expected - {axis.upper()} calculated (mm)")

            plt.hist2d(df[f"{axis.upper()}_exp"], df[f"{axis.upper()}_diff"], bins=40, cmap='YlOrRd')
            plt.ylim(-4, 4)
            plt.colorbar(label='Frequency')
            plt.grid(True)
            plt.show()
        else:
            # Used to display the difference between expected and obtained positions in the given direction, with error bars (standard deviation)

            # Calculate the average and standard deviation of axis_diff grouped by axis_exp
            grouped = df.groupby(f'{axis.upper()}_exp')[f'{axis.upper()}_diff']
            average_diff = grouped.mean()
            std_diff = grouped.std()

            # Plotting the scatter graph with error bars
            plt.figure(figsize=(10, 6)) # Increased figure size for better readability
            plt.title(f'Average {axis.upper()}_diff vs {axis.upper()}_exp ({independent_var.upper()}_exp = 0) with Error Bars (std)', fontsize=14)  # Increased fontsize
            plt.xlabel(f'{axis.upper()} Expected (mm)', fontsize=12) # Increased fontsize
            plt.ylabel(f'Average {axis.upper()}_diff (mm)', fontsize=12) # Increased fontsize

            plt.errorbar(average_diff.index, average_diff.valuesf, yerr=std_diff.values,
                        fmt='o', color='blue', capsize=5, label=f'Average {axis.upper()}_diff') # Added error bars

            plt.grid(True)
            plt.legend() 
            plt.tight_layout()
            plt.show()


    def rows_columns_yield(self, row_column):
        # Read the data from CSV file
        file_path = self.file   # 'C:/Users/Felipe Casarin/Desktop/Incidence-Position-On-Phoswich-Detector/collection_files_csv_unfiltered/-6_-6.csv' 

        df = pd.read_csv(file_path, usecols=[f"{row_column}"])

        # Filtering data to remove zeroes and outliers
        filtered_data=df[df[f"{row_column}"]<3000]
        filtered_data=filtered_data[filtered_data[f"{row_column}"]>0]

        # Plotting the heatmap using plt.hist2d()
        plt.figure(figsize=(8, 6))
        plt.title(f"{row_column} Yield")
        plt.xlabel('Yield')
        plt.ylabel('Number of events')

        plt.hist(filtered_data[f"{row_column}"], bins=50)
        plt.grid(True)
        plt.show()
    
    def yx_pos_distribution(self, x, y):
        # Read the data from Csv
        file_path = self.path    # 'C:/Users/Felipe Casarin/Desktop/Incidence-Position-On-Phoswich-Detector/diff_values_0903_pixdim3.csv' 

        df = pd.read_csv(file_path, usecols=['X', 'Y', 'X_exp', 'Y_exp'])

        axis = self.axis.lower()
        independent_var = self.independent_var.lower()

        filtered_data=df[df['X_exp']== x]
        filtered_data=filtered_data[filtered_data['Y_exp']== y]

        # Plotting the heatmap using plt.hist2d()
        plt.figure(figsize=(8, 6))
        plt.title(f'Distribution of estimated positions in the {axis.upper()}-axis for the data point ({x},{y})')
        plt.xlabel(f'{axis.upper()} (mm)')
        plt.ylabel('Number of events')

        plt.hist(filtered_data[f'{axis.upper()}'], bins=50)
        plt.grid(True)
        plt.show()

    def YX_heatmap(self, is_single_plot, x=0, y=0):
        file_path = self.path + "/" + x + "_" + y + "_pos.xlsx"   # 'C:/Users/Felipe Casarin/Desktop/Incidence-Position-On-Phoswich-Detector/outfiles_pos_calc_0903_pixdim3/2_0_pos.xlsx' 

        df = pd.read_excel(file_path, usecols=['X', 'Y'])

        if is_single_plot:

            plt.figure(figsize=(8, 6))
            plt.title(f'({x},{y}) arrange')
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.hist2d(df['X'], df['Y'], bins=20, cmap='YlOrRd')
            plt.colorbar(label='Frequency')
            plt.grid(True)
            plt.show()
        
        else:
            xtab=np.array([-6, -4, -2, 0, 2, 4, 6])
            ytab=np.array([-6, -4, -2, 0, 2, 4])

            for x in xtab:
                for y in ytab:
                    file_path = self.path
                    file_path = f'{file_path}/{x}_{y}_pos.xlsx' 



                    df = pd.read_excel(file_path, usecols=['X', 'Y'])

                    plt.figure(figsize=(8, 6))
                    plt.title(f'({x},{y}) arrange')
                    plt.xlabel('X (mm)')
                    plt.ylabel('Y (mm)')

                    plt.hist2d(df['X'], df['Y'], bins=20, cmap='YlOrRd')
                    plt.colorbar(label='Frequency')
                    plt.grid(True)

                    # Show the plots
                    save_path = os.path.join(f"{file_path}/position_images/", f"{x}_{y}_distribution.png")
                    plt.savefig(save_path, dpi=300)
                    plt.close() 