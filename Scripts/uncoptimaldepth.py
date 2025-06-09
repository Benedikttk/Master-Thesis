import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import process_file, calculate_fractions

data_folder = r'C:\Users\benja\Desktop\Speciale\FilesForUnc'

# Create empty lists to store full file paths
be10_files = []
b10_files = []

# Loop over all files in the folder
for filename in sorted(os.listdir(data_folder)):
    if filename.startswith("Be10_run") and filename.endswith(".txt"):
        be10_files.append(os.path.join(data_folder, filename))
    elif filename.startswith("B10_run") and filename.endswith(".txt"):
        b10_files.append(os.path.join(data_folder, filename))

# Select first file pair
be10_file = be10_files[0]
b10_file = b10_files[0]

# Process files
df_Be10 = process_file(be10_file, 'Be')
df_B10 = process_file(b10_file, 'B')

# Convert depth to µm (for plotting/consistency)
df_Be10['Depth (µm)'] = df_Be10['Depth (Angstrom)'] / 1e4
df_B10['Depth (µm)'] = df_B10['Depth (Angstrom)'] / 1e4

# Total ion counts
total_ions_Be10 = df_Be10['Be Ions'].sum()
total_ions_B10 = df_B10['B Ions'].sum()

# Optimization loop
best_cutoff_depth = df_B10[df_B10['B Ions'] > 0]['Depth (Angstrom)'].max()
best_ratio = 0

for depth in np.linspace(df_B10['Depth (Angstrom)'].min(),
                         df_B10['Depth (Angstrom)'].max(),
                         len(df_B10)):

    Be10_fraction, B10_fraction = calculate_fractions(
        depth, df_Be10, df_B10, total_ions_Be10, total_ions_B10)

    if B10_fraction < 1 and B10_fraction <= 0.95:
        ratio = Be10_fraction / (1 - B10_fraction)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cutoff_depth = depth

# Final results with optimal cutoff
Be10_fraction, B10_fraction = calculate_fractions(
    best_cutoff_depth, df_Be10, df_B10, total_ions_Be10, total_ions_B10)

# Output results
print("---------- Optimal Cutoff for First File ----------")
print(f"Optimized Cutoff Depth: {best_cutoff_depth:.2f} Å")
print(f"Be10 fraction beyond cutoff: {Be10_fraction * 100:.2f}%")
print(f"B10 fraction beyond cutoff: {(1 - B10_fraction) * 100:.2f}%")


# List to store optimal cutoff depths and file names
best_cutoff_depths = []
used_files = []

# Loop over all matched Be10 and B10 files
for be10_file, b10_file in zip(be10_files, b10_files):
    # Process files
    df_Be10 = process_file(be10_file, 'Be')
    df_B10 = process_file(b10_file, 'B')

    # Add depth in µm
    df_Be10['Depth (µm)'] = df_Be10['Depth (Angstrom)'] / 1e4
    df_B10['Depth (µm)'] = df_B10['Depth (Angstrom)'] / 1e4

    # Total ion counts
    total_ions_Be10 = df_Be10['Be Ions'].sum()
    total_ions_B10 = df_B10['B Ions'].sum()

    # Optimization
    best_cutoff_depth = df_B10[df_B10['B Ions'] > 0]['Depth (Angstrom)'].max()
    best_ratio = 0

    for depth in np.linspace(df_B10['Depth (Angstrom)'].min(),
                             df_B10['Depth (Angstrom)'].max(),
                             len(df_B10)):
        Be10_fraction, B10_fraction = calculate_fractions(
            depth, df_Be10, df_B10, total_ions_Be10, total_ions_B10)

        if B10_fraction < 1 and B10_fraction <= 0.95:
            ratio = Be10_fraction / (1 - B10_fraction)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cutoff_depth = depth

    # Store results
    best_cutoff_depths.append(best_cutoff_depth)
    used_files.append((be10_file, b10_file))

    # Print result for this pair
    print(f"Processed:\n  Be10 file: {os.path.basename(be10_file)}\n  B10 file:  {os.path.basename(b10_file)}")
    print(f"  → Best cutoff depth: {best_cutoff_depth:.2f} Å\n")

# Summary
print("✔ Finished processing all files.")



cutoff_array = np.array(best_cutoff_depths) / 1e4  # Convert Å to µm
mean_cutoff = np.mean(cutoff_array)
std_cutoff = np.std(cutoff_array, ddof=1)  # sample standard deviation
sem_cutoff = std_cutoff / np.sqrt(len(cutoff_array))

print(f"Optimized cutoff depth (mean ± SEM): {mean_cutoff:.4f} µm ± {sem_cutoff:.4f} µm")


import os
import numpy as np
import matplotlib.pyplot as plt
from Functions import process_file

folder_unc = r"C:\Users\benja\Desktop\Speciale\FilesForUnc"

all_files = os.listdir(folder_unc)

# Filter and sort B files starting with B and ending .txt (e.g. B_run1.txt)
b_files = sorted([f for f in all_files if f.startswith('B') and f.endswith('.txt')])

# Filter and sort Be files starting with Be and ending .txt (e.g. Be_run21.txt)
be_files = sorted([f for f in all_files if f.startswith('Be') and f.endswith('.txt')])

# Assuming pairing by order: first Be file pairs with first B file, etc.
if len(be_files) != len(b_files):
    print(f"Warning: Different number of Be ({len(be_files)}) and B ({len(b_files)}) files.")

plt.figure(figsize=(12, 7))

for be_file, b_file in zip(be_files, b_files):
    path_be = os.path.join(folder_unc, be_file)
    path_b = os.path.join(folder_unc, b_file)
    
    df_be = process_file(path_be, 'Be')
    df_b = process_file(path_b, 'B')
    
    df_be['Depth (µm)'] = df_be['Depth (Angstrom)'] / 1e4
    df_b['Depth (µm)'] = df_b['Depth (Angstrom)'] / 1e4
    
    plt.hist(df_be['Depth (µm)'], bins=len(df_be), weights=df_be['Be Ions'], 
             histtype='step', linewidth=1.5, label=f"{be_file} (Be)", alpha=0.7)
    
    plt.hist(df_b['Depth (µm)'], bins=len(df_b), weights=df_b['B Ions'], 
             histtype='step', linewidth=1.5, linestyle='--', label=f"{b_file} (B)", alpha=0.7)

plt.xlabel('Depth (µm)')
plt.ylabel('Ion Count')
plt.title('Depth Distributions of Be and B Ions (All Files)')
plt.legend(fontsize=8, loc='upper right', ncol=2)
#plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
