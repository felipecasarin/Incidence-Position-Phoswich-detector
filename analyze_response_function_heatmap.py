import matplotlib.pyplot as plt
import pandas as pd

# Load data
file_path = "outfiles_no_filter/diff_allcolumns.csv"
df = pd.read_csv(file_path)

# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Heatmap for X-axis
hb1 = axes[0].hexbin(df["cL1"], df["X"], gridsize=50, cmap="Blues", mincnt=1)
axes[0].scatter(df["cL1"], df["X_exp"], color="black", s=1, alpha=0.5)  # Expected positions
axes[0].set_title("Hexbin: Estimated vs Expected X")
axes[0].set_xlabel("L1 Light Yield")
axes[0].set_ylabel("X Position (mm)")
fig.colorbar(hb1, ax=axes[0])

# Heatmap for Y-axis
hb2 = axes[1].hexbin(df["cL1"], df["Y"], gridsize=50, cmap="Reds", mincnt=1)
axes[1].scatter(df["cL1"], df["Y_exp"], color="black", s=1, alpha=0.5)  # Expected positions
axes[1].set_title("Hexbin: Estimated vs Expected Y")
axes[1].set_xlabel("L1 Light Yield")
axes[1].set_ylabel("Y Position (mm)")
fig.colorbar(hb2, ax=axes[1])

plt.tight_layout()
plt.show()
