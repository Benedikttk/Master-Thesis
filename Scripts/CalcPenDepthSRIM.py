import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem  # For standard error of the mean
from scipy.optimize import curve_fit  # For curve fitting
from Functions import process_file, calculate_fractions  # Import calculate_fractions



file_path_Be10 = r"C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\RANGE_1400_ion_1000Be10.txt"
file_path_B10 = r"C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\RANGE_1400_ion_1000B10.txt"

df_SRIM_depth_Be10 = process_file(file_path_Be10, 'Be')
df_SRIM_depth_B10 = process_file(file_path_B10, 'B')

# Calculate histograms
hist_Be10, bins_Be10 = np.histogram(df_SRIM_depth_Be10['Depth (Angstrom)'], 
                                    bins=len(df_SRIM_depth_Be10['Depth (Angstrom)']), 
                                    weights=df_SRIM_depth_Be10['Be Ions'])

hist_B10, bins_B10 = np.histogram(df_SRIM_depth_B10['Depth (Angstrom)'], 
                                  bins=len(df_SRIM_depth_B10['Depth (Angstrom)']), 
                                  weights=df_SRIM_depth_B10['B Ions'])

# Plotting the two distributions
plt.figure(figsize=(10, 6))

plt.hist(df_SRIM_depth_Be10['Depth (Angstrom)'], 
         bins=len(df_SRIM_depth_Be10['Depth (Angstrom)']),
         weights=df_SRIM_depth_Be10['Be Ions'],
         color='red', alpha=0.6, histtype='stepfilled', label=r'$\mathrm{^{10}Be}$')

plt.hist(df_SRIM_depth_B10['Depth (Angstrom)'],
         bins=len(df_SRIM_depth_B10['Depth (Angstrom)']),
         weights=df_SRIM_depth_B10['B Ions'],
         color='blue', alpha=0.4, histtype='stepfilled', label=r'$\mathrm{^{10}B}$')

# Making the cutoff for B10
mask = df_SRIM_depth_B10['B Ions'] > 0
df_SRIM_depth_B10 = df_SRIM_depth_B10[mask]
cutoff_depth = df_SRIM_depth_B10['Depth (Angstrom)'].max()
plt.axvline(x=cutoff_depth, color='black', linestyle='--', label=r'Cutoff for $\mathrm{^{10}B}$')

# Calculate the fraction of Be10 and B10 that penetrates the cutoff
total_ions_Be10 = df_SRIM_depth_Be10['Be Ions'].sum()
total_ions_B10 = df_SRIM_depth_B10['B Ions'].sum()

# Use the calculate_fractions function from Functions, passing total_ions_Be10 and total_ions_B10
Be10_fraction, B10_fraction = calculate_fractions(cutoff_depth, df_SRIM_depth_Be10, df_SRIM_depth_B10, total_ions_Be10, total_ions_B10)

# Optimize the cutoff depth to maximize the fraction of Be10 ions that penetrate while minimizing the fraction of B10 ions that penetrate
best_cutoff_depth = cutoff_depth
best_ratio = 0
for depth in np.linspace(df_SRIM_depth_B10['Depth (Angstrom)'].min(), df_SRIM_depth_B10['Depth (Angstrom)'].max(), len(df_SRIM_depth_B10['Depth (Angstrom)'])):
    Be10_fraction, B10_fraction = calculate_fractions(depth, df_SRIM_depth_Be10, df_SRIM_depth_B10, total_ions_Be10, total_ions_B10)
    if B10_fraction < 1:
        ratio = Be10_fraction / (1 - B10_fraction)
        if B10_fraction <= 0.95:
            if ratio > best_ratio:
                best_ratio = ratio
                best_cutoff_depth = depth

# Recalculate fractions with the optimized cutoff depth
Be10_fraction, B10_fraction = calculate_fractions(best_cutoff_depth, df_SRIM_depth_Be10, df_SRIM_depth_B10, total_ions_Be10, total_ions_B10)

# Plot the optimized cutoff
plt.axvline(x=best_cutoff_depth, color='green', linestyle='--', label='Optimized Cutoff')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(direction="in", length=6, which="major")  # Major ticks longer
plt.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
plt.minorticks_on()
billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'


# Plotting the two distributions
plt.xlabel('Depth (Angstrom)')
plt.ylabel('Ion Count')
plt.title(r'Simulated Depth Distribution of $\mathrm{^{10}Be}$ and $\mathrm{^{10}B}$ Ions in SiN (SRIM/TRIM)')
plt.legend()
plt.tight_layout()

# Calculate the total count of particles to the right of the optimized cutoff
total_count_after_cutoff = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > best_cutoff_depth]['Be Ions'].sum() + \
                           df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] > best_cutoff_depth]['B Ions'].sum()

# Calculate the fraction of Be10 and B10 ions to the right of the optimized cutoff
Be10_after_cutoff_fraction = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > best_cutoff_depth]['Be Ions'].sum() / total_count_after_cutoff
B10_after_cutoff_fraction = df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] > best_cutoff_depth]['B Ions'].sum() / total_count_after_cutoff

print('----------------Information-------------------')
print(f"The cutoff depth for B10 is {cutoff_depth:.2f} Angstrom")
print(f"The fraction of Be10 ions that penetrates the cutoff is {Be10_fraction*100:.2f}%")
print(f"The fraction of B10 ions that is stopped before the cutoff is {B10_fraction*100:.2f}%")
print(f"Optimized cutoff depth: {best_cutoff_depth:.2f} Angstrom")
print(f"The fraction of Be10 ions that penetrates the optimized cutoff is {Be10_fraction*100:.2f}%")
print(f"The fraction of B10 ions that penetrates the optimized cutoff is {100-B10_fraction*100:.2f}%")
print(f"Total count of particles to the right of the optimized cutoff: {total_count_after_cutoff}")
print(f"Fraction of particles to the right of the optimized cutoff that are Be10: {Be10_after_cutoff_fraction*100:.2f}%")
print(f"Fraction of particles to the right of the optimized cutoff that are B10: {B10_after_cutoff_fraction*100:.2f}%")

with open('BoronsupressionDepthIndo.txt', 'w') as file:
    file.write('----------------Information-------------------\n')
    file.write(f"The cutoff depth for B10 is {cutoff_depth:.2f} Angstrom\n")
    file.write(f"The fraction of Be10 ions that penetrates the cutoff is {Be10_fraction*100:.2f}%\n")
    file.write(f"The fraction of B10 ions that is stopped before the cutoff is {B10_fraction*100:.2f}%\n")
    file.write(f"Optimized cutoff depth: {best_cutoff_depth:.2f} Angstrom\n")
    file.write(f"The fraction of Be10 ions that penetrates the optimized cutoff is {Be10_fraction*100:.2f}%\n")
    file.write(f"The fraction of B10 ions that penetrates the optimized cutoff is {100-B10_fraction*100:.2f}%\n")
    file.write(f"Total count of particles to the right of the optimized cutoff: {total_count_after_cutoff}\n")
    file.write(f"Fraction of particles to the right of the optimized cutoff that are Be10: {Be10_after_cutoff_fraction*100:.2f}%\n")
    file.write(f"Fraction of particles to the right of the optimized cutoff that are B10: {B10_after_cutoff_fraction*100:.2f}%\n")




plt.show()


l = [16068.75, 15900.00, 16350.00, 15985.72, 16114.29, 16671.43, 16478.57, 16600.00, 16800.00, 16800.00, 17053.85]
energy = [900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000]
plt.figure(figsize=(10, 6))
j = [17215.39]
mev = [1011.1]
plt.plot(energy, l, marker='o', linestyle='-', color='red')
plt.plot(mev, j, marker='o', linestyle='-', color='blue')
plt.xlabel('Energy (keV)')
plt.ylabel('Penetration Depth (Angstrom)')
plt.show()