import numpy as np
import matplotlib.pyplot as plt
import os
from Functions import count_valid_ions, read_exyz_file

#test
import matplotlib as mpl

# Example: match LaTeX document font size of 12pt
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})



# Directory containing data files
data_dir = r"C:\Users\benja\Desktop\Speciale\Nydata\Be(1.38)@SiN(1.86)"
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
            
            # Define constants for each file
            effective_length = 3.15e9  # angstrom (315 mm)
            anode_1_length = effective_length / 3  # delta E

            # Count valid ions and calculate average length
            valid_ions, avg_length, uncertainty = count_valid_ions(df, effective_length, anode_1_length)
            valid_ion_counts.append(valid_ions)
            avg_lengths.append(avg_length)
            uncertainties.append(uncertainty)
            
            # Write results to file
            file.write(f"Density: {density}, Valid Ions: {valid_ions}, Avg Length: {avg_length}, Uncertainty: ±{uncertainty}\n")

# Convert density (g/cm3) → pressure (Pa)
T = 293.15  # Kelvin
R = 8.3145  # J/mol·K
M = 0.05812  # kg/mol (isobutane)
density_values_kg_m3 = np.array(density_values) * 1000  # g/cm3 → kg/m3
pressure_values = (density_values_kg_m3 * R * T) / M  # Pressure in Pascals
pressure_values /= 100  # Convert to mbar

print(f'Presure values {pressure_values}')
# Sort by density for plotting
if density_values:
    sorted_indices = np.argsort(density_values)
    density_values = np.array(density_values)[sorted_indices]
    valid_ion_counts = np.array(valid_ion_counts)[sorted_indices]
    avg_lengths = np.array(avg_lengths)[sorted_indices] #
    uncertainties = np.array(uncertainties)[sorted_indices]

    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Number of Valid Ions vs. Gas Density
    ax1 = axs[0]
     #ax1.plot(pressure_values, valid_ion_counts, 'o--', markersize=4, label='Valid Ions')
    ion_count_err = np.array([1.7378657930901624, 1.8986314748727977, 1.8819139193916392, 1.7829983188274505, 1.7654908735791748, 1.7728073654651202, 1.7788735197353829, 1.7578781912535688, 1.7398971064448292, 1.7428403716347065, 1.7257552817583977, 1.657036733454687, 1.6469407967875742, 1.213644419963388, 1.3093073414159542])

    ax1.errorbar(pressure_values, valid_ion_counts, yerr=ion_count_err, fmt='o--', markersize=4, capsize=5, capthick=2, label = "SIM Data")
    ax1.fill_between(
    pressure_values,
    valid_ion_counts - ion_count_err,
    valid_ion_counts + ion_count_err,
    color='red',
    alpha=0.2,  # adjust transparency
    label='±1 sigma band'
)
    ax1.set_xlabel(r'Gas Pressure (mbar)')
    #ax1.plot(density_values, valid_ion_counts, 'o--', markersize=4, label='Valid Ions')
    #ax1.set_xlabel(r'Gas Density ($g/cm^{3}$)')
    ax1.set_ylabel('Number of Valid Ions')
    ax1.set_title(r'Number of Valid $\mathrm{^{10}Be}$ Ions vs. Gas Density')
    ax1.grid(True)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(direction="in", length=6, which="major")  # Major ticks longer
    ax1.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
    ax1.minorticks_on()

    # Plot 2: Average Ion Length vs. Gas Density with Error Bars
    ax2 = axs[1]
    #ax2.errorbar(density_values, avg_lengths, yerr=uncertainties, fmt='o--', markersize=4, capsize=5, capthick=2, label='Average Ion Length')
    #ax2.set_xlabel(r'Gas Density ($g/cm^{3}$)')
    # Plot 2
    range_err = np.array([
  430478453.1131315,
  477001284.9570629,
  471396912.4410977,
  470724207.4572427,
  444749715.7419476,
  417180622.5556163,
  378808906.5866653,
  343125837.43529594,
  315061703.4409201,
  291454343.69137007,
  270631092.8517649,
  187055701.8559745,
  112290053.59049739,
  65324963.001870245,
  28605337.0216547
])*1/1e9
    ax2.errorbar(pressure_values, avg_lengths*1/1e8, yerr=range_err, fmt='o--', markersize=4, capsize=5, capthick=2, label = "SIM Data") #Å->cm
    ax2.fill_between(
    pressure_values,
    (avg_lengths*1/1e8) - range_err,
    (avg_lengths*1/1e8) + range_err,
    color='red',
    alpha=0.2,  # adjust transparency
    label='±1 sigma band'
    )
    ax2.set_xlabel(r'Gas Pressure (mbar)')
    ax2.set_ylabel('Average Ion Length (cm)')
    ax2.set_title('Average Ion Track Length vs. Gas Density')
    ax2.grid(True)
    ax2.legend()

    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(direction="in", length=6, which="major")  # Major ticks longer
    ax2.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
    ax2.minorticks_on()

    # Save plot to file before showing it
    #billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'
    billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
    plt.tight_layout()
    plt.savefig(f'{billeder_path}\\SRIM_OptimizedGasDensity_SIMS.pdf')

    # Now display the plot
    plt.show()

else:
    print("No valid files found.")

print(avg_lengths/1e8)

