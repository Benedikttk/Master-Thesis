import numpy as np
from Functions import calculate_ratio_and_efficiency

Be51_830kV_counts = 20001
Be51_830kV_runs = 2*6 
avg_Be51_counts = Be51_830kV_counts/Be51_830kV_runs

Be9_current = 2.19297*1e-006

detector_live_time = 49.545

Be9cnts = (Be9_current * detector_live_time) / 1.6e-19




R_n, R_n_uncertainty, iso_eff, iso_eff_uncertainty = calculate_ratio_and_efficiency(avg_Be51_counts, Be9cnts, np.sqrt(avg_Be51_counts), 27.1e-12, 0.3e-12, runs=Be51_830kV_runs)

print(f"The isotropic ratio efficiency is {round(iso_eff, 3)} ± {round(iso_eff_uncertainty, 3)} %")


