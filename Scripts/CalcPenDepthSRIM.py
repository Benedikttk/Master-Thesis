import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem  # For standard error of the mean
from scipy.optimize import curve_fit  # For curve fitting
from Functions import process_file, calculate_fractions  # Import calculate_fractions


#test
import matplotlib as mpl

# Example: match LaTeX document font size of 12pt
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})



#file_path_Be10 = r"C:\Users\benja\Desktop\Speciale\Data\RANGE_1400_ion_1000Be10.txt"
#file_path_B10 = r"C:\Users\benja\Desktop\Speciale\Data\RANGE_1400_ion_1000B10.txt"

file_path_Be10 = r"C:\Users\benja\Desktop\noge\10000.txt"
file_path_100B10 = r"C:\Users\benja\Desktop\noge\b100k.txt"


df_SRIM_depth_Be10 = process_file(file_path_Be10, 'Be')
df_SRIM_depth_B10 = process_file(file_path_100B10, 'B')


df_SRIM_depth_Be10['Depth (µm)'] = df_SRIM_depth_Be10['Depth (Angstrom)'] / 1e4
df_SRIM_depth_B10['Depth (µm)'] = df_SRIM_depth_B10['Depth (Angstrom)'] / 1e4

# Calculate histograms
hist_Be10, bins_Be10 = np.histogram(df_SRIM_depth_Be10['Depth (µm)'], 
                                    bins=len(df_SRIM_depth_Be10['Depth (µm)']), 
                                    weights=df_SRIM_depth_Be10['Be Ions'])

hist_B10, bins_B10 = np.histogram(df_SRIM_depth_B10['Depth (µm)'], 
                                  bins=len(df_SRIM_depth_B10['Depth (µm)']), 
                                  weights=df_SRIM_depth_B10['B Ions'])

# Plotting the two distributions
plt.figure(figsize=(10, 6), )

plt.hist(df_SRIM_depth_Be10['Depth (µm)'], 
         bins=len(df_SRIM_depth_Be10['Depth (µm)']),
         weights=df_SRIM_depth_Be10['Be Ions'],
         color='red', alpha=0.6, histtype='stepfilled', label=r'$\mathrm{^{10}Be}$')

plt.hist(df_SRIM_depth_B10['Depth (µm)'],
         bins=len(df_SRIM_depth_B10['Depth (µm)']),
         weights=df_SRIM_depth_B10['B Ions'],
         color='blue', alpha=0.4, histtype='stepfilled', label=r'$\mathrm{^{10}B}$')

# Making the cutoff for B10
mask = df_SRIM_depth_B10['B Ions'] > 0
df_SRIM_depth_B10 = df_SRIM_depth_B10[mask]
cutoff_depth = df_SRIM_depth_B10['Depth (µm)'].max()
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
plt.axvline(x=best_cutoff_depth/10000, color='green', linestyle='--', label='Optimised Cutoff')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(direction="in", length=6, which="major")  # Major ticks longer
plt.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
plt.minorticks_on()
billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'


# Plotting the two distributions
plt.xlabel(r'Depth ($\mu$m)')
plt.ylabel('Ion Count')
plt.title(r'Simulated Depth Distribution of $\mathrm{^{10}Be}$ and $\mathrm{^{10}B}$ Ions in $Si_{3}N_{4}$ TRIM')
plt.yscale('log')
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



#plt.text(18630, 20000, 'B10 Cutoff (18600 Å)', rotation=90, verticalalignment='bottom')
#plt.text(17245, 20000, 'Optimised Cutoff (17215 Å)', rotation=90, color='green', verticalalignment='bottom')

'''textstr = '\n'.join((
    r'$^{10}Be$ penetration: 98.7%',
    r'$^{10}B$ stopped: 92.9%',
    'Optimised cutoff: 93.29% $^{10}Be$, 6.71% $^{10}B$'
))

plt.annotate(textstr, xy=(500/10000, 35000), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

'''
#extra stuff
maximumdepthB10 = np.max(df_SRIM_depth_B10['Depth (Angstrom)'])
print("test", maximumdepthB10)



billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
#billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'
plt.savefig(f'{billeder_path}\\BoronSupressionDepth.pdf')
plt.show()

print(np.mean(df_SRIM_depth_B10['Depth (Angstrom)'])/1e4)


B_max_sep = [17400.0, 18300.0, 18300.0, 18300.0, 18300.0, 18300.0]
B_optimal_sep = []
#B10be10_ratio = ["1:1", 2000/100, 4000/100, 6000/100, 8000/100, 10000/100]
B10be10_ratio = ["1:10", "1:100", "1:1000", "1:2000", "1:4000", "1:6000"]

plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(direction="in", length=6, which="major")  # Major ticks longer
plt.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
plt.minorticks_on()
plt.scatter(B10be10_ratio, B_max_sep)

plt.show()