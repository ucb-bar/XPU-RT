import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the provided CSV file
file_path = "data/runtime.csv"  # Update with the correct path if necessary
data = pd.read_csv(file_path)

# Extract the columns for plotting
num_jobs = data["num_jobs"]
num_machines = data["num_machines"]
runtime = data["runtime"]

# Create a grid for plotting
grid_jobs, grid_machines = np.meshgrid(
    np.unique(num_jobs), np.unique(num_machines)
)
grid_runtime = np.zeros_like(grid_jobs, dtype=float)

# Fill the grid with runtime values
for i, (job, machine, run) in enumerate(zip(num_jobs, num_machines, runtime)):
    x_idx = np.where(np.unique(num_jobs) == job)[0][0]
    y_idx = np.where(np.unique(num_machines) == machine)[0][0]
    grid_runtime[y_idx, x_idx] = run

# Create the 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a log scale for runtime
surf = ax.plot_surface(
    grid_jobs, grid_machines, np.log10(grid_runtime),
    cmap='viridis', edgecolor='k', alpha=0.8
)

# Adjust the view
# ax.view_init(elev=30, azim=45)  # Elevation (up/down) and azimuthal angle (rotation)

# Add labels and title
ax.set_xlabel("Number of Operations")
ax.set_ylabel("Number of Machines")
ax.set_zlabel("Log10 Runtime (s)")
ax.set_title("Runtime Surface (Log Scale)")

# Add color bar with log scale
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Log10 Runtime (s)")

# Save the plot with the rotated view
output_file = "plots/runtime_surface.pdf"
plt.savefig(output_file, dpi=500, bbox_inches='tight')
plt.show()

print(f"Plot saved to {output_file}")