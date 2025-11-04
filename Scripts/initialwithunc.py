import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_exyz_file(file_path):
    column_names = [
        "Ion Number", "Energy (keV)", "Depth (X) (Angstrom)", 
        "Y (Angstrom)", "Z (Angstrom)", "Electronic Stop.(eV/A)", 
        "Energy lost due to Last Recoil(eV)"
    ]
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=15, names=column_names)
    # Replace commas with dots for decimals and convert to numeric
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    return df

def count_valid_ions_per_ion(df, effective_length, anode_1_length):
    valid_ion_counts = {}
    ion_ranges = {}
    for ion_number in sorted(df["Ion Number"].unique()):
        ion_data = df[df["Ion Number"] == ion_number]
        x_positions = ion_data["Depth (X) (Angstrom)"]

        # Calculate ion range as max - min depth
        ion_range = x_positions.max() - x_positions.min()
        ion_ranges[ion_number] = ion_range

        # Check validity conditions on ion depth positions
        if (x_positions.max() >= anode_1_length and
            x_positions.max() <= effective_length and
            x_positions.min() <= anode_1_length):
            valid_ion_counts[ion_number] = len(ion_data)
        else:
            valid_ion_counts[ion_number] = 0
    return valid_ion_counts, ion_ranges

# Path to your folder with data files
folder_path = r"C:\Users\benja\Desktop\Speciale\Nydata\Be(1.38)@SiN(1.86)"

effective_length = 3.15e9  # angstrom
anode_1_length = effective_length / 3

file_names = []
official_avg_valid_ions_list = []
std_valid_steps_list = []
official_avg_range_list = []
std_range_list = []

for file in os.listdir(folder_path):
    if file.endswith(".csv.txt") or file.endswith(".txt") or file.endswith(".csv"):
        full_path = os.path.join(folder_path, file)

        df = read_exyz_file(full_path)
        valid_ion_counts, ion_ranges = count_valid_ions_per_ion(df, effective_length, anode_1_length)

        # Only consider ions with valid counts > 0
        valid_counts = [count for count in valid_ion_counts.values() if count > 0]
        valid_ranges = [ion_ranges[ion_num] for ion_num, count in valid_ion_counts.items() if count > 0]

        # Compute averages and std devs on valid ions only
        average_valid_steps = np.mean(valid_counts) if valid_counts else 0
        std_valid_steps = np.std(valid_counts) if valid_counts else 0
        average_range = np.mean(valid_ranges) if valid_ranges else 0
        std_range = np.std(valid_ranges) if valid_ranges else 0

        file_names.append(file)
        official_avg_valid_ions_list.append(average_valid_steps)
        std_valid_steps_list.append(std_valid_steps)
        official_avg_range_list.append(average_range)
        std_range_list.append(std_range)

        print(f"{file}:")
        print(f"  Average valid steps per ion: {average_valid_steps:.2f}")
        print(f"  Std dev of valid steps per ion: {std_valid_steps:.2f}")
        print(f"  Average ion range (Angstrom): {average_range:.2f}")
        print(f"  Std dev of ion range (Angstrom): {std_range:.2f}")
        print()

# Summary printout
print("All average valid steps per file:", official_avg_valid_ions_list)
print("All std dev of valid steps per file:", std_valid_steps_list)
print("All average ion ranges per file:", official_avg_range_list)
print("All std dev of ion ranges per file:", std_range_list)
