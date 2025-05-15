import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem  # For standard error of the mean
from scipy.optimize import curve_fit  # For curve fitting
from Functions import process_file, calculate_fractions  # Import calculate_fractions
import matplotlib as mpl

avg_exit_energy = 146.43147567476717 #KeV @ avg_range 1.5150003300000001
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
df_SRIM_depth_Be10['Depth (micro)'] = df_SRIM_depth_Be10['Depth (Angstrom)']/1e4

df_SRIM_depth_B10 = process_file(file_path_100B10, 'B')
df_SRIM_depth_B10['Depth (micro)'] = df_SRIM_depth_B10['Depth (Angstrom)']/1e4

# Calculate histograms
hist_Be10, bins_Be10 = np.histogram(df_SRIM_depth_Be10['Depth (micro)'], 
                                    bins=len(df_SRIM_depth_Be10['Depth (micro)']), 
                                    weights=df_SRIM_depth_Be10['Be Ions'])

hist_B10, bins_B10 = np.histogram(df_SRIM_depth_B10['Depth (micro)'], 
                                    bins=len(df_SRIM_depth_B10['Depth (micro)']), 
                                    weights=df_SRIM_depth_B10['B Ions'])



# 1. Define a Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# 2. Prepare histogram data for fitting
bin_centers = 0.5 * (bins_B10[:-1] + bins_B10[1:])  # Compute bin centers

# Remove bins with zero counts for fitting (optional but can help)
nonzero = hist_B10 > 0
x_fit = bin_centers[nonzero]
y_fit = hist_B10[nonzero]

# 3. Initial guess for parameters: A, mu, sigma
initial_guess = [np.max(y_fit), np.mean(x_fit), np.std(x_fit)]

# 4. Fit the Gaussian to the data
popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=initial_guess)

# Extract fitted parameters
A_fit, mu_fit, sigma_fit = popt
print(f"Fitted parameters:\n  A = {A_fit:.2f}\n  mu = {mu_fit:.2f} Å\n  sigma = {sigma_fit:.2f} Å")

# 5. Plot the histogram and the fitted Gaussian
plt.figure(figsize=(10, 6))

plt.hist(df_SRIM_depth_B10['Depth (micro)'], 
         bins=len(df_SRIM_depth_B10['Depth (micro)']),
         weights=df_SRIM_depth_B10['B Ions'],
         color='red', alpha=0.4, histtype='stepfilled', label=r'$\mathrm{^{10}B}$ data')

x_smooth = np.linspace(min(x_fit), max(x_fit), 1000)
plt.plot(x_smooth, gaussian(x_smooth, *popt), 'k--', label='Gaussian fit')

plt.xlabel('Depth (my m)')
plt.ylabel('Ion Count (weighted)')
plt.title('Gaussian Fit to Ion Depth Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

dE_dx = 1286.65 #KeV/mym
total_ions = df_SRIM_depth_B10['B Ions'].sum()

print(f'Spread in avg energy {abs(dE_dx)*sigma_fit/(np.sqrt(total_ions))}')

print(f'Exit energy of B @ 1.38 MeV in {np.mean(df_SRIM_depth_Be10['Depth (micro)'])} my m of SiN: {245.65} +- {abs(dE_dx)*sigma_fit/(np.sqrt(total_ions))}')