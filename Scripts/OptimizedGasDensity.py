from EXYZReader import read_exyz_file
import numpy as np
import matplotlib.pyplot as plt
import os

# Define constants
effective_length = 3.15e9  # angstrom (315 mm)
anode_1_length = effective_length / 3  # delta E
anode_2_length = 2 * effective_length / 3  # E_res

# Function to count valid ions and calculate average length
def count_valid_ions(df):
    valid_ions = 0
    max_lengths = []  # Store max lengths of valid ions
    for ion_number in range(1, max(df["Ion Number"]) + 1):
        ion_data = df[df["Ion Number"] == ion_number]
        x_positions = ion_data["Depth (X) (Angstrom)"]
        
        # Check if ion crosses first anode and reaches second anode without surpassing effective length
        if (x_positions.max() >= anode_1_length and 
            x_positions.max() <= effective_length and 
            x_positions.min() <= anode_1_length):
            valid_ions += 1
            max_lengths.append(x_positions.max())  # Store max length of valid ion
    
    # Calculate average length and uncertainty (standard deviation)
    if max_lengths:
        avg_length = np.mean(max_lengths)
        uncertainty = np.std(max_lengths)
    else:
        avg_length = 0
        uncertainty = 0
    
    return valid_ions, avg_length, uncertainty

# Directory containing data files
data_dir = r"C:\Users\benja\Desktop\Speciale\Data\EXYZs\Denistydata"
density_values = []
valid_ion_counts = []
avg_lengths = []
uncertainties = []
with open('OptimizedGadDensityInfo.txt', 'w') as file:
# Loop over different density files
    for file_name in os.listdir(data_dir):
        if file_name.startswith("GIC_detector_data_density_") and file_name.endswith(".csv.txt"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_name}")
            
            # Extract density value from file name
            density_str = file_name.split('_')[-1].replace('.csv.txt', '').replace(',', '.')
            density = float(density_str)
            density_values.append(density)
            
            # Read the data file
            df = read_exyz_file(file_path)
            
            # Count valid ions and calculate average length
            valid_ions, avg_length, uncertainty = count_valid_ions(df)
            valid_ion_counts.append(valid_ions)
            avg_lengths.append(avg_length)
            uncertainties.append(uncertainty)
            file.write(f"Density: {density}, Valid Ions: {valid_ions}, Avg Length: {avg_length}, Uncertainty: ±{uncertainty}\n")

# Sort by density for plotting
if density_values:
    sorted_indices = np.argsort(density_values)
    density_values = np.array(density_values)[sorted_indices]
    valid_ion_counts = np.array(valid_ion_counts)[sorted_indices]
    avg_lengths = np.array(avg_lengths)[sorted_indices]
    uncertainties = np.array(uncertainties)[sorted_indices]

    # Plot 1: Number of Valid Ions vs. Gas Density
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(density_values, valid_ion_counts, 'o-', label='Valid Ions')
    plt.xlabel('Gas Density (g/cm³)')
    plt.ylabel('Number of Valid Ions')
    plt.title('Optimal Gas Density for GIC Detector')
    plt.grid(True)
    plt.legend()

    # Plot 2: Average Ion Length vs. Gas Density with Error Bars
    plt.subplot(1, 2, 2)
    plt.errorbar(density_values, avg_lengths, yerr=uncertainties, fmt='o-', label='Average Ion Length')
    plt.xlabel('Gas Density (g/cm³)')
    plt.ylabel('Average Ion Length (Å)')
    plt.title('Average Ion Length vs. Gas Density')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No valid files found.")