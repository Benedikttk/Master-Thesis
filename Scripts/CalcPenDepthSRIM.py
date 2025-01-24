import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem  # For standard error of the mean
from scipy.optimize import curve_fit # For curve fitting
from ReadingFromSRIMfile import process_file  

#file_path_Be10 = r"C:\Users\benja\Desktop\Speciale\Data\RANGE_1400_ion_1000Be10.txt"
#file_path_B10 = r"C:\Users\benja\Desktop\Speciale\Data\RANGE_1400_ion_1000B10.txt"

file_path_Be10 = r"C:\Users\benja\Desktop\Speciale\Data\RANGE_2400_be.txt"
file_path_B10 = r"C:\Users\benja\Desktop\Speciale\Data\RANGE.txt"




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
         color='red', alpha=0.6, histtype='stepfilled', label='Be10')

plt.hist(df_SRIM_depth_B10['Depth (Angstrom)'],
         bins=len(df_SRIM_depth_B10['Depth (Angstrom)']),
         weights=df_SRIM_depth_B10['B Ions'],
         color='blue', alpha=0.4, histtype='stepfilled', label='B10')

# Making the cutoff for B10
mask = df_SRIM_depth_B10['B Ions'] > 0
df_SRIM_depth_B10 = df_SRIM_depth_B10[mask]
cutoff_depth = df_SRIM_depth_B10['Depth (Angstrom)'].max()
plt.axvline(x=cutoff_depth, color='black', linestyle='--', label='Cutoff for B10')

# Calculate the fraction of Be10 and B10 that penetrates the cutoff
total_ions_Be10 = df_SRIM_depth_Be10['Be Ions'].sum()
total_ions_B10 = df_SRIM_depth_B10['B Ions'].sum()

Be10_after_cutoff = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > cutoff_depth]['Be Ions'].sum()
B10_before_cutoff = df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] <= cutoff_depth]['B Ions'].sum()

Be10_fraction = Be10_after_cutoff / total_ions_Be10
B10_fraction = B10_before_cutoff / total_ions_B10

print('----------------Information-------------------')
print(f"The cutoff depth for B10 is {cutoff_depth:.2f} Angstrom")
print(f"The fraction of Be10 ions that penetrates the cutoff is {Be10_fraction*100:.2f}%")
print(f"The fraction of B10 ions that is stopped before the cutoff is {B10_fraction*100:.2f}%")

# Function to calculate the fraction of Be10 and B10 ions based on a given cutoff depth
def calculate_fractions(cutoff_depth, df_SRIM_depth_Be10, df_SRIM_depth_B10):
    Be10_after_cutoff = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > cutoff_depth]['Be Ions'].sum()
    B10_before_cutoff = df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] <= cutoff_depth]['B Ions'].sum()
    
    Be10_fraction = Be10_after_cutoff / total_ions_Be10
    B10_fraction = B10_before_cutoff / total_ions_B10
    
    return Be10_fraction, B10_fraction

# Optimize the cutoff depth to maximize the fraction of Be10 ions that penetrate while minimizing the fraction of B10 ions that penetrate
best_cutoff_depth = cutoff_depth
best_ratio = 0
for depth in np.linspace(df_SRIM_depth_B10['Depth (Angstrom)'].min(), df_SRIM_depth_B10['Depth (Angstrom)'].max(), len(df_SRIM_depth_B10['Depth (Angstrom)'])):
    Be10_fraction, B10_fraction = calculate_fractions(depth, df_SRIM_depth_Be10, df_SRIM_depth_B10)
    if B10_fraction < 1:
        ratio = Be10_fraction / (1 - B10_fraction)
        if B10_fraction <= 0.95:
            if ratio > best_ratio:
                best_ratio = ratio
                best_cutoff_depth = depth

# Recalculate fractions with the optimized cutoff depth
Be10_fraction, B10_fraction = calculate_fractions(best_cutoff_depth, df_SRIM_depth_Be10, df_SRIM_depth_B10)

print(f"Optimized cutoff depth: {best_cutoff_depth:.2f} Angstrom")
print(f"The fraction of Be10 ions that penetrates the optimized cutoff is {Be10_fraction*100:.2f}%")
print(f"The fraction of B10 ions that penetrates the optimized cutoff is {100-B10_fraction*100:.2f}%")

# Plot the optimized cutoff
plt.axvline(x=best_cutoff_depth, color='green', linestyle='--', label='Optimized Cutoff')

# Plotting the two distributions
plt.xlabel('Depth (Angstrom)')
plt.ylabel('Ion Count')
plt.title('Depth Distribution of Be10 and B10 Ions')
plt.legend()
plt.tight_layout()

# Calculate the total count of particles to the right of the optimized cutoff
total_count_after_cutoff = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > best_cutoff_depth]['Be Ions'].sum() + \
                           df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] > best_cutoff_depth]['B Ions'].sum()

# Calculate the fraction of Be10 and B10 ions to the right of the optimized cutoff
Be10_after_cutoff_fraction = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > best_cutoff_depth]['Be Ions'].sum() / total_count_after_cutoff
B10_after_cutoff_fraction = df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] > best_cutoff_depth]['B Ions'].sum() / total_count_after_cutoff

print(f"Total count of particles to the right of the optimized cutoff: {total_count_after_cutoff}")
print(f"Fraction of particles to the right of the optimized cutoff that are Be10: {Be10_after_cutoff_fraction*100:.2f}%")
print(f"Fraction of particles to the right of the optimized cutoff that are B10: {B10_after_cutoff_fraction*100:.2f}%")

plt.show()
