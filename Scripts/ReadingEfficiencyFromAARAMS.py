import numpy as np
from Functions import get_txt_files, read_block_data, parse_dataframe, calculate_Be10_statistics, calculate_Be10_current, extract_metadata, calculate_Be9_ions, calculate_ratio_and_efficiency
import matplotlib.pyplot as plt
import seaborn as sns

# Path
filepath = r'C:\Users\benja\Desktop\Speciale\NyBeeffdata'
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
        avg_Be10cnts, Be9cnts, std_Be10cnts, 27.1e-12, 0.3e-12, runs=120)

    print(f"The ratio of Be10/Be9 is {R_n} ± {R_n_uncertainty}")
    print(f"The isotropic ratio efficiency is {round(iso_eff, 3)} ± {round(iso_eff_uncertainty, 3)} %")

# Set Be10 counts to 1404.9
avg_Be10cnts = 1891.4

# Original measured data
counting_algorithm = np.array([1404.9, 1891.4, 1056.783, 149.341])
counting_algorithm_err = np.array([43.49, 37.48, 32.5082,12.2])

counting_normal = np.array([1400.5, 1886.4, 1052.44, 148.76])
counting_normal_err = np.array([108.33, 57.80, 194.381, 12.98])

# Estimate Poisson parameters
lambda_algorithm = np.mean(counting_algorithm)
lambda_normal = np.mean(counting_normal)

# Monte Carlo Simulations
n_samples = 10000
simulated_algorithm = np.random.poisson(lam=lambda_algorithm, size=n_samples)
simulated_normal = np.random.poisson(lam=lambda_normal, size=n_samples)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define positions
box_positions = [1, 2]
violin_positions = [1.3, 2.3]

# Create box plot (updated tick_labels parameter)
ax.boxplot([counting_algorithm, counting_normal], positions=box_positions, widths=0.3,
           patch_artist=True, tick_labels=["Algorithm", "Normal"],
           boxprops=dict(facecolor='skyblue', color='black', alpha=0.6),
           whiskerprops=dict(color='black'),
           capprops=dict(color='black'),
           medianprops=dict(color='black', linewidth=2))

# Create x positions for error bars
x_algorithm = np.full_like(counting_algorithm, 1, dtype=float)
x_normal = np.full_like(counting_normal, 2, dtype=float)

# Add error bars (fixed)
ax.errorbar(x_algorithm, counting_algorithm, yerr=counting_algorithm_err, fmt='o', color='red',
            label='Algorithm Errors', markersize=6, capsize=5, elinewidth=2)
ax.errorbar(x_normal, counting_normal, yerr=counting_normal_err, fmt='o', color='blue',
            label='Normal Errors', markersize=6, capsize=5, elinewidth=2)

# Violin plots
parts = ax.violinplot([simulated_algorithm, simulated_normal], positions=violin_positions, widths=0.2, showmeans=False, showmedians=True)

# Customize violin plots
for pc in parts['bodies']:
    pc.set_facecolor('gray')
    pc.set_alpha(0.4)

# Labels and title
ax.set_ylabel('Counting Values', fontsize=14)
ax.set_title('Box Plot with Simulated Distributions', fontsize=16)

# Update x-axis ticks
ax.set_xticks([1, 1.3, 2, 2.3])
ax.set_xticklabels(["Algorithm", "Sim. Algorithm", "Normal", "Sim. Normal"], rotation=15)

# Grid and styling
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(axis='y', which='both', direction='in', length=6, width=1, colors='black', grid_color='gray', grid_alpha=0.5)
ax.tick_params(axis='y', which='minor', direction='in', length=4, width=1, colors='black')

# Legend
ax.legend(loc='best')

# Save figure
billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
plt.savefig(f'{billeder_path}\\comparisonofalgorithms.pdf', dpi=300, bbox_inches='tight')

# Show plot
plt.show()


plt.figure(figsize=(10, 6))

# Histogram (or KDE)
sns.histplot(simulated_algorithm, kde=True, color='red', stat='density', label='Simulated Algorithm', bins=30, alpha=0.6)
sns.histplot(simulated_normal, kde=True, color='blue', stat='density', label='Simulated Normal', bins=30, alpha=0.6)

# Optional: Mean lines
plt.axvline(np.mean(counting_algorithm), color='darkred', linestyle='--', label='Algorithm Mean')
plt.axvline(np.mean(counting_normal), color='darkblue', linestyle='--', label='Normal Mean')

plt.xlabel('Count Values')
plt.ylabel('Density')
plt.title('Simulated Distributions of Counting Methods')
plt.legend()
plt.grid(True)

# Save if needed
# plt.savefig(f'{billeder_path}\\simulated_distributions.pdf', dpi=300, bbox_inches='tight')

plt.show()