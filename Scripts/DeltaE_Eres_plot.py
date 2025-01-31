from EXYZReader import read_exyz_file
import numpy as np
import matplotlib.pyplot as plt

# Define constants
effective_length = 3.15e9  # angstrom (315 mm)
anode_1_length = effective_length / 3  # delta E
anode_2_length = 2 * effective_length / 3  # E_res

# Function to calculate Delta E and E_res for valid ions
def calculate_delta_e_e_res(df):
    delta_e_values = []
    e_res_values = []
    
    for ion_number in range(1, max(df["Ion Number"]) + 1):
        ion_data = df[df["Ion Number"] == ion_number]
        x_positions = ion_data["Depth (X) (Angstrom)"]
        energies = ion_data["Energy (keV)"]
        
        # Check if ion crosses first anode, reaches second anode, and does not exceed effective length
        if (x_positions.max() >= anode_2_length and  # Must reach anode 2
            x_positions.max() <= effective_length and  # Must not exceed effective length
            x_positions.min() <= anode_1_length):  # Must pass anode 1
            
            # Calculate Delta E (energy loss in the first anode)
            delta_e = energies.iloc[0] - energies.iloc[np.where(x_positions >= anode_1_length)[0][0]]
            
            # Calculate E_res (residual energy in the second anode)
            e_res_index = np.where(x_positions >= anode_2_length)[0][0]
            e_res = energies.iloc[e_res_index]
            
            # Ensure E_res is not zero
            if e_res > 0:
                delta_e_values.append(delta_e)
                e_res_values.append(e_res)
    
    return delta_e_values, e_res_values

# File paths for Boron and Beryllium data
boron_file_path = r"C:\Users\benja\Desktop\Speciale\Data\DeltaE_Eres\EXYZ_B_2200keV_14000,39.txt"
beryllium_file_path = r"C:\Users\benja\Desktop\Speciale\Data\DeltaE_Eres\EXYZ_Be_1400keV_14000,39.txt"

# Process Boron data
print(f"Processing Boron file: {boron_file_path}")
df_boron = read_exyz_file(boron_file_path)
boron_delta_e, boron_e_res = calculate_delta_e_e_res(df_boron)

# Process Beryllium data
print(f"Processing Beryllium file: {beryllium_file_path}")
df_beryllium = read_exyz_file(beryllium_file_path)
beryllium_delta_e, beryllium_e_res = calculate_delta_e_e_res(df_beryllium)

# Plot Delta E vs E_res for Boron and Beryllium
plt.figure(figsize=(10, 8))
plt.scatter(boron_e_res, boron_delta_e, alpha=0.5, label="Boron", color="blue")
plt.scatter(beryllium_e_res, beryllium_delta_e, alpha=0.5, label="Beryllium", color="red")
plt.xlabel('Residual Energy (E_res) [keV]')
plt.ylabel('Energy Loss (Delta E) [keV]')
plt.title('Delta E vs E_res for Boron and Beryllium')
plt.legend()
billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
plt.savefig(f'{billeder_path}\\deltaE_Eres.pdf')

plt.show()