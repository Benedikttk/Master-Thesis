import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_srim_stopping_table(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for idx, line in enumerate(lines):
        if "Ion" in line and "Lateral" in line:
            header_idx = idx
            break
    
    data_lines = lines[header_idx + 3:]
    data = []

    for line in data_lines:
        if '-----------' in line or not line.strip():
            break
        parts = line.split()
        if len(parts) >= 4:
            Ion_Energy_str, Ion_Energy_unit, dEdx_elec_str, dEdx_nuc_str, projected_range_str, projected_range_unit = parts[:6]

            Ion_Energy = float(Ion_Energy_str.replace(',', '.'))
            Ion_Energy_unit = Ion_Energy_unit.strip("'")

            projected_range = float(projected_range_str.replace(',', '.'))
            projected_range_unit = projected_range_unit.strip("'")

            if Ion_Energy_unit == "eV":
                Ion_Energy /= 1000
            elif Ion_Energy_unit == "MeV":
                Ion_Energy *= 1000

            if projected_range_unit == "A":
                projected_range /= 1e4  # convert Å to µm
            elif projected_range_unit == "um":
                pass  # already in µm
            elif projected_range_unit == "mm":
                projected_range *= 1e3  # mm to µm

            dEdx_elec = float(dEdx_elec_str.replace(',', '.'))
            dEdx_nuc = float(dEdx_nuc_str.replace(',', '.'))

            data.append([Ion_Energy, dEdx_elec, dEdx_nuc, projected_range])
        
    df = pd.DataFrame(data, columns=['Ion Energy', 'dE/dx elec.', 'dE/dx Nuclear', 'Projected Range'])
    return df

# Load files
be_file = r'C:\Users\benja\Desktop\Speciale\Nydata\isobutansimsSRIM\Beryllium in Iso-Butane (ICRU-493) (gas) - Kopi.txt'
b_file  = r'C:\Users\benja\Desktop\Speciale\Nydata\extra\BoroninHC (gas).txt'

be_df = process_srim_stopping_table(be_file)
b_df  = process_srim_stopping_table(b_file)

def get_range_at_energy(df, E_exit):
    closest = df.iloc[(df['Ion Energy'] - E_exit).abs().argmin()]
    return closest['Ion Energy'], closest['Projected Range']

# Example exit energies from foil (keV) — change as needed per foil thickness
E_Be_exit = 743.6
E_B_exit = 245.7


E_found_Be, R_Be_1atm_um = get_range_at_energy(be_df, E_Be_exit)
E_found_B, R_B_1atm_um = get_range_at_energy(b_df, E_B_exit)

# Convert ranges from µm to cm
R_Be_1atm = R_Be_1atm_um / 1e4
R_B_1atm = R_B_1atm_um / 1e4

print(f"\nUsing E_Be_exit = {E_Be_exit:.2f} keV → Range = {R_Be_1atm:.5f} cm")
print(f"Using E_B_exit  = {E_B_exit:.2f} keV → Range = {R_B_1atm:.5f} cm\n")

def calculate_pressure_window(R_B_1atm_cm, R_Be_1atm_cm, L_total_cm):
    L1 = L_total_cm / 3
    L2 = 2 * L_total_cm / 3

    p_B_min = R_B_1atm_cm / (L1 * 0.9)
    p_Be_min = R_Be_1atm_cm / (L1 + L2)
    p_Be_max = R_Be_1atm_cm / (L1 * 1.1)

    p_min = max(p_B_min, p_Be_min)
    p_max = p_Be_max
    return p_min, p_max

L_total = 31.0  # cm, total detector length
p_min, p_max = calculate_pressure_window(R_B_1atm, R_Be_1atm, L_total)

print(f"Valid pressure window (atm): {p_min:.5f} to {p_max:.5f}")
print(f"Equivalent in mbar: {p_min * 1013:.2f} to {p_max * 1013:.2f}")
