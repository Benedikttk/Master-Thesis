import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define file path
filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'

# Subject headers for identifying data sections
subject = "[CDAT0"
names = ["dE", "E final"]

# List all relevant files
list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(".txt.mpa")]
if not list_of_raw_files:
    raise ValueError("No valid data files found!")

print("Found raw files:", list_of_raw_files)

# Function to extract data from a file
def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        print(f"Warning: {file_path} is empty!")
        return None, None, None

    # Find header indices
    header_indices = []
    for idx, line in enumerate(lines):
        if any(f"NAME={name}" in line and line.strip() == f"NAME={name}" for name in names):
            header_indices.append(idx)

    if len(header_indices) < 2:
        print(f"Skipping {file_path} (missing headers)")
        return None, None, None

    # Extract dE data
    dE_values, dE_counts = extract_section(lines, header_indices[0])

    # Extract E_final data
    E_final_values, E_final_counts = extract_section(lines, header_indices[1])

    if dE_values is None or E_final_values is None:
        print(f"Skipping {file_path} (incomplete data)")
        return None, None, None

    # Merge into a DataFrame
    df = pd.DataFrame({'E_final': E_final_values, 'dE': dE_values, 'counts': dE_counts})
    df_dE = pd.DataFrame({'dE': dE_values, 'counts': dE_counts})
    df_E_final = pd.DataFrame({'E_final': E_final_values, 'counts': E_final_counts})

    return df, df_dE, df_E_final

# Function to extract a [DATA] section
def extract_section(lines, start_idx):
    section_lines = lines[start_idx:]
    data_start = next((i for i, line in enumerate(section_lines) if "[DATA]" in line), None)

    if data_start is None:
        return None, None

    # Read data, stopping at the first non-numeric line
    data = []
    for line in section_lines[data_start + 1:]:
        if any(char.isalpha() for char in line):
            break
        data.append(line.strip().split())

    if not data:
        return None, None

    # Convert to numeric
    df = pd.DataFrame(data, columns=["value", "counts"])
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    df["counts"] = pd.to_numeric(df["counts"], errors='coerce')
    df.dropna(inplace=True)
    
    return df["value"].values, df["counts"].values

# Process first valid file (for simplicity)
for file in list_of_raw_files:
    df, df_dE, df_E_final = extract_data(os.path.join(filepath, file))
    if df is not None:
        break  # Stop after first successful extraction

if df is None:
    raise ValueError("No valid data extracted!")

# 🔥 **Plot 1: Scatter plot (E_final vs. dE, colored by counts)**
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(df["E_final"], df["dE"], c=df["counts"], cmap='viridis', s=1)
plt.xlabel("E_final")
plt.ylabel("dE")
plt.colorbar(label="Counts")
plt.title("Scatter: E_final vs dE")

# 🔥 **Plot 2: dE vs. Counts**
plt.subplot(1, 3, 2)
plt.plot(df_dE["dE"], df_dE["counts"], label="dE", color="blue")
plt.xlabel("dE")
plt.ylabel("Counts")
plt.title("dE vs Counts")
plt.legend()

# 🔥 **Plot 3: E_final vs. Counts**
plt.subplot(1, 3, 3)
plt.plot(df_E_final["E_final"], df_E_final["counts"], label="E_final", color="red")
plt.xlabel("E_final")
plt.ylabel("Counts")
plt.title("E_final vs Counts")
plt.legend()

plt.tight_layout()
plt.show()
