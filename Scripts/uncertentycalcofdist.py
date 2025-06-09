import os
import numpy as np
import matplotlib.pyplot as plt
from Functions import process_file

# Path to Be10 files
folder = r'C:\Users\benja\Desktop\Speciale\FilesForUnc'

# Store mean projected ranges
mean_ranges = []

# Loop over files
for file in sorted(os.listdir(folder)):
    if file.startswith('Be10_run') and file.endswith('.txt'):
        filepath = os.path.join(folder, file)
        df = process_file(filepath, 'Be')

        # Convert to µm
        df['Depth (µm)'] = df['Depth (Angstrom)'] / 1e4

        # Weighted average depth (projected range)
        depths = df['Depth (µm)'].values
        counts = df['Be Ions'].values
        mean_range = np.average(depths, weights=counts)
        mean_ranges.append(mean_range)

# Convert to numpy array
mean_ranges = np.array(mean_ranges)

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.hist(mean_ranges, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Mean Projected Range (µm)')
plt.ylabel('Number of Simulations')
plt.title(r'Distribution of Mean Projected Range of $\mathrm{^{10}Be}$ Ions')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Print summary
print(f"Mean of means: {np.mean(mean_ranges):.4f} µm")
print(f"Standard deviation: {np.std(mean_ranges, ddof=1):.4f} µm")
print(f"Standard error: {np.std(mean_ranges, ddof=1)/np.sqrt(len(mean_ranges)):.4f} µm")
