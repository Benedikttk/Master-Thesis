import numpy as np
import pandas as pd
from config import *
from data_reader import read_srim_file, read_exyz_file
from calculations import calculate_fractions, calculate_delta_e_e_res
from plotting import plot_depth_distribution, plot_delta_e_e_res

# Example usage
if __name__ == "__main__":
    # Read SRIM data
    df_srim_depth_Be10 = read_srim_file(rf"{DATA_DIR}\RANGE_1400_ion_1000Be10.txt", 'Be')
    df_srim_depth_B10 = read_srim_file(rf"{DATA_DIR}\RANGE_1400_ion_1000B10.txt", 'B')

    # Calculate fractions and plot
    cutoff_depth = df_srim_depth_B10['Depth (Angstrom)'].max()
    Be10_fraction, B10_fraction = calculate_fractions(cutoff_depth, df_srim_depth_Be10, df_srim_depth_B10)
    plot_depth_distribution(df_srim_depth_Be10, df_srim_depth_B10, cutoff_depth, cutoff_depth, BILLEDER_PATH)

    # Read EXYZ data
    df_boron = read_exyz_file(rf"{DATA_DIR}\DeltaE_Eres\EXYZ_B_2200keV_14000,39.txt")
    df_beryllium = read_exyz_file(rf"{DATA_DIR}\DeltaE_Eres\EXYZ_Be_1400keV_14000,39.txt")

    # Calculate Delta E and E_res
    boron_delta_e, boron_e_res = calculate_delta_e_e_res(df_boron, ANODE_1_LENGTH, ANODE_2_LENGTH, EFFECTIVE_LENGTH)
    beryllium_delta_e, beryllium_e_res = calculate_delta_e_e_res(df_beryllium, ANODE_1_LENGTH, ANODE_2_LENGTH, EFFECTIVE_LENGTH)

    # Plot Delta E vs E_res
    plot_delta_e_e_res(boron_e_res, boron_delta_e, beryllium_e_res, beryllium_delta_e, BILLEDER_PATH)