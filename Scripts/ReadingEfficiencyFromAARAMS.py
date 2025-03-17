import numpy as np
from Functions import get_txt_files, read_block_data, parse_dataframe, calculate_Be10_statistics, calculate_Be10_current, extract_metadata, calculate_Be9_ions, calculate_ratio_and_efficiency
import matplotlib.pyplot as plt
import seaborn as sns


#path
filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'

list_of_files = get_txt_files(filepath, ".txt")

number_of_index = 0

if not list_of_files:
    print("No .txt files found.")
else:
    data_lines = read_block_data(filepath + "\\" + list_of_files[number_of_index])
    
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
    
    detector_live_time = extract_metadata(filepath + "\\" + list_of_files[number_of_index], "Detector live time [s]")
    Be9_current = extract_metadata(filepath + "\\" + list_of_files[number_of_index], "9Be current [A]")
    
    print(f"The detector live time is: {detector_live_time} [s]")
    print(f"The current of Be9 is: {Be9_current} [A]")
    
    Be9cnts = calculate_Be9_ions(Be9_current, detector_live_time)
    print(f"The number of Be9 ions is {Be9cnts}")
    
    R_n, R_n_uncertainty, iso_eff, iso_eff_uncertainty = calculate_ratio_and_efficiency(
        avg_Be10cnts, Be9cnts, std_Be10cnts, 27.1e-12, 0.3e-12)
    
    print(f"The ratio of Be10/Be9 is {R_n} ± {R_n_uncertainty}")
    print(f"The isotropic ratio efficiency is {round(iso_eff, 3)} ± {round(iso_eff_uncertainty, 3)} %")
    
    
  # Set Be10 counts to 1404.9
avg_Be10cnts = 1891.4

#The sum of the filtered_roi_cluster_points for Be10 is: 1891.4 ± 43.49022878762539
# 1404.9 ± 37.481995677925156

# Make sure Be9cnts is correctly calculated
Be9cnts = calculate_Be9_ions(Be9_current, detector_live_time)

# Now call the function to calculate the ratio and efficiency
R_n, R_n_uncertainty, iso_eff, iso_eff_uncertainty = calculate_ratio_and_efficiency(
    avg_Be10cnts, Be9cnts, std_Be10cnts, 27.1e-12, 0.3e-12
)

# Output the results
print(f"The ratio of Be10/Be9 is {R_n} ± {R_n_uncertainty}")
print(f"The isotropic ratio efficiency, using algorithm, is {round(iso_eff, 3)} ± {round(iso_eff_uncertainty, 3)} %")


# Original measured data
counting_algorithm = np.array([1404.9, 1891.4])
counting_algorithm_err = np.array([43.49, 37.48])

counting_normal = np.array([1400.5, 1886.4])
counting_normal_err = np.array([108.33, 57.80])

# Estimate Poisson parameters (mean of measured data)
lambda_algorithm = np.mean(counting_algorithm)
lambda_normal = np.mean(counting_normal)

# Monte Carlo Simulations
n_samples = 10000
simulated_algorithm = np.random.poisson(lam=lambda_algorithm, size=n_samples)
simulated_normal = np.random.poisson(lam=lambda_normal, size=n_samples)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define positions for box plots
box_positions = [1, 2]  # Measured data positions
violin_positions = [1.3, 2.3]  # Slightly offset positions for simulated data

# Create box plot for measured data
ax.boxplot([counting_algorithm, counting_normal], positions=box_positions, widths=0.3,
           patch_artist=True, labels=["Algorithm", "Normal"],
           boxprops=dict(facecolor='skyblue', color='black', alpha=0.6),
           whiskerprops=dict(color='black'),
           capprops=dict(color='black'),
           medianprops=dict(color='black', linewidth=2))

# Add error bars for measured values
ax.errorbar([1, 1], counting_algorithm, yerr=counting_algorithm_err, fmt='o', color='red', 
             label='Algorithm Errors', markersize=6, capsize=5, elinewidth=2)
ax.errorbar([2, 2], counting_normal, yerr=counting_normal_err, fmt='o', color='blue', 
             label='Normal Errors', markersize=6, capsize=5, elinewidth=2)

# Overlay simulated distributions using Matplotlib violin plots
parts = ax.violinplot([simulated_algorithm, simulated_normal], positions=violin_positions, widths=0.2, showmeans=False, showmedians=True)

# Customize violin plots
for pc in parts['bodies']:
    pc.set_facecolor('gray')
    pc.set_alpha(0.4)

# Labels and title
ax.set_ylabel('Counting Values', fontsize=14)
ax.set_title('Box Plot with Simulated Distributions', fontsize=16)

# Adjust x-axis ticks to align with measured data and simulated data
ax.set_xticks([1, 1.3, 2, 2.3])
ax.set_xticklabels(["Algorithm", "Sim. Algorithm", "Normal", "Sim. Normal"], rotation=15)

# Add grid and custom styling
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)  # Apply grid only to the y-axis
ax.minorticks_on()  # Enable minor ticks for y-axis only
ax.tick_params(axis='y', which='both', direction='in', length=6, width=1, colors='black', grid_color='gray', grid_alpha=0.5)  # Major ticks on y-axis
ax.tick_params(axis='y', which='minor', direction='in', length=4, width=1, colors='black')  # Minor ticks on y-axis


# Add legend at the best location
ax.legend(loc='best')

# Corrected file path
billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'

# Save the figure as a PDF file
plt.savefig(f'{billeder_path}\\comparisonofalgorithms.pdf', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()

# Calculate the average isotropic ratio (just for reference)
isotropic_efficiency = np.array([15.885, 15.935, 16.201, 16.244])
isotropic_efficiency_err = np.array([0.215, 0.215, 0.186, 0.187])

# Average isotropic ratio and its uncertainty
average_isotropic = np.mean(isotropic_efficiency)
average_isotropic_err = np.sqrt(np.sum(isotropic_efficiency_err**2)) / len(isotropic_efficiency)

average_isotropic, average_isotropic_err


print(f"The average isotropic value{average_isotropic} ± {average_isotropic_err}")