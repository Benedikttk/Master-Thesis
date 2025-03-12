import os
import pandas as pd
import numpy as np
from Functions import get_txt_files, read_block_data, parse_dataframe, calculate_Be10_statistics, calculate_Be10_current, extract_metadata, calculate_Be9_ions, calculate_ratio_and_efficiency

#path
filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'

list_of_files = get_txt_files(filepath, ".txt")

if not list_of_files:
    print("No .txt files found.")
else:
    data_lines = read_block_data(filepath + "\\" + list_of_files[0])
    
    column_names = [
        "Blk", "10Becnts", "Totalcnts", "10Bcnts", "cnts", "cnts",
        "LiveTime", "10Becps", "9Becur", "nonecur", "9BeOcur", "nonecur",
        "10Be/9Be", "10Be/none", "9Be/none", "9Be/9BeO", "none/none",
        "9BeO/none", "TargetCur", "Flags"
    ]
    
    df = parse_dataframe(data_lines, column_names)
    
    avg_Be10cnts, std_Be10cnts = calculate_Be10_statistics(df)
    print(f"The average Be10 counts is {avg_Be10cnts} and the standard deviation is ± {std_Be10cnts}")
    
    avg_time = df["LiveTime"].astype(float).mean()
    time_uncertainty = df["LiveTime"].astype(float).std()
    
    I_Be10, I_Be10_uncertainty = calculate_Be10_current(avg_Be10cnts, avg_time, time_uncertainty)
    print(f"The current of Be10 is {I_Be10} ± {I_Be10_uncertainty} [micro A]")
    
    detector_live_time = extract_metadata(filepath + "\\" + list_of_files[0], "Detector live time [s]")
    Be9_current = extract_metadata(filepath + "\\" + list_of_files[0], "9Be current [A]")
    
    print(f"The detector live time is: {detector_live_time} [s]")
    print(f"The current of Be9 is: {Be9_current} [A]")
    
    Be9cnts = calculate_Be9_ions(Be9_current, detector_live_time)
    print(f"The number of Be9 ions is {Be9cnts}")
    
    R_n, R_n_uncertainty, iso_eff, iso_eff_uncertainty = calculate_ratio_and_efficiency(
        avg_Be10cnts, Be9cnts, std_Be10cnts, 27.1e-12, 0.3e-12)
    
    print(f"The ratio of Be10/Be9 is {R_n} ± {R_n_uncertainty}")
    print(f"The isotropic ratio efficiency is {round(iso_eff, 3)} ± {round(iso_eff_uncertainty, 3)} %")
    
    
  # Set Be10 counts to 1404.9
avg_Be10cnts = 1404.9

# Make sure Be9cnts is correctly calculated
Be9cnts = calculate_Be9_ions(Be9_current, detector_live_time)

# Now call the function to calculate the ratio and efficiency
R_n, R_n_uncertainty, iso_eff, iso_eff_uncertainty = calculate_ratio_and_efficiency(
    avg_Be10cnts, Be9cnts, std_Be10cnts, 27.1e-12, 0.3e-12
)

# Output the results
print(f"The ratio of Be10/Be9 is {R_n} ± {R_n_uncertainty}")
print(f"The isotropic ratio efficiency is {round(iso_eff, 3)} ± {round(iso_eff_uncertainty, 3)} %")
