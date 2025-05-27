import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import sqrt, log
from Functions import process_file

# --- Plot settings ---
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})

# --- File paths ---
file_path_Be10 = r"C:\Users\benja\Desktop\Speciale\noge\Be100.txt"
file_path_100B10 = r"C:\Users\benja\Desktop\Speciale\noge\B100.txt"

# --- Load and process data ---
df_SRIM_depth_Be10 = process_file(file_path_Be10, 'Be')
df_SRIM_depth_B10 = process_file(file_path_100B10, 'B')

df_SRIM_depth_Be10['Depth (µm)'] = df_SRIM_depth_Be10['Depth (Angstrom)'] / 1e4
df_SRIM_depth_B10['Depth (µm)'] = df_SRIM_depth_B10['Depth (Angstrom)'] / 1e4

# --- Histogram and binning ---
bins = 100
hist_Be10, bins_Be10 = np.histogram(df_SRIM_depth_Be10['Depth (µm)'], bins=bins, weights=df_SRIM_depth_Be10['Be Ions'])
hist_B10, bins_B10 = np.histogram(df_SRIM_depth_B10['Depth (µm)'], bins=bins, weights=df_SRIM_depth_B10['B Ions'])

bin_centers_Be10 = (bins_Be10[:-1] + bins_Be10[1:]) / 2
bin_centers_B10 = (bins_B10[:-1] + bins_B10[1:]) / 2

# --- Gaussian fitting ---
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Fit Be-10
popt_Be10, _ = curve_fit(gaussian, bin_centers_Be10, hist_Be10,
                         p0=[np.max(hist_Be10), bin_centers_Be10[np.argmax(hist_Be10)], 0.01])
amp_Be10, mean_Be10, std_Be10 = popt_Be10
fwhm_Be10 = 2 * sqrt(2 * log(2)) * std_Be10
half_max_Be10 = amp_Be10 / 2

# Fit B-10
popt_B10, _ = curve_fit(gaussian, bin_centers_B10, hist_B10,
                        p0=[np.max(hist_B10), bin_centers_B10[np.argmax(hist_B10)], 0.01])
amp_B10, mean_B10, std_B10 = popt_B10
fwhm_B10 = 2 * sqrt(2 * log(2)) * std_B10
half_max_B10 = amp_B10 / 2

# --- Print results ---
print(f"Be-10 Fit: Mean = {mean_Be10:.4f} µm, Std Dev = {std_Be10:.4f} µm, FWHM = {fwhm_Be10:.4f} µm")
print(f"B-10 Fit: Mean = {mean_B10:.4f} µm, Std Dev = {std_B10:.4f} µm, FWHM = {fwhm_B10:.4f} µm")
# --- Plotting with fig, ax ---
fig, ax = plt.subplots(figsize=(6, 6))

ax.grid(True, linestyle='--', alpha=0.6)
ax.minorticks_on()
ax.tick_params(direction="in", length=6, which="major")
ax.tick_params(direction="in", length=3, which="minor")

# Histograms
ax.hist(df_SRIM_depth_Be10['Depth (µm)'], bins=bins, weights=df_SRIM_depth_Be10['Be Ions'],
        color='lightcoral', alpha=0.5, histtype='stepfilled', label=r'$\mathrm{^{10}Be}$')
ax.hist(df_SRIM_depth_B10['Depth (µm)'], bins=bins, weights=df_SRIM_depth_B10['B Ions'],
        color='skyblue', alpha=0.4, histtype='stepfilled', label=r'$\mathrm{^{10}B}$')

# Gaussian curves
x_fit = np.linspace(min(bin_centers_Be10.min(), bin_centers_B10.min()),
                    max(bin_centers_Be10.max(), bin_centers_B10.max()), 1000)

ax.plot(x_fit, gaussian(x_fit, *popt_Be10), color='darkred', linestyle='--', linewidth=2,
        label=fr'$^{{10}}$Be Fit (μ={mean_Be10:.3f} µm)')
ax.plot(x_fit, gaussian(x_fit, *popt_B10), color='darkblue', linestyle='--', linewidth=2,
        label=fr'$^{{10}}$B Fit (μ={mean_B10:.3f} µm)')

# Draw FWHM lines
ax.hlines(half_max_Be10, mean_Be10 - fwhm_Be10 / 2, mean_Be10 + fwhm_Be10 / 2,
          colors='darkred', linestyles='-', linewidth=2)
ax.hlines(half_max_B10, mean_B10 - fwhm_B10 / 2, mean_B10 + fwhm_B10 / 2,
          colors='darkblue', linestyles='-', linewidth=2)

# Text annotations for FWHM
ax.text(mean_Be10 + 0.13, amp_Be10 * 0.5, fr'FWHM: {fwhm_Be10:.3f} µm', color='darkred')
ax.text(mean_B10 - 0.83, amp_B10 * 0.5, fr'FWHM: {fwhm_B10:.3f} µm', color='darkblue')

# Labels and title
ax.set_xlabel("Depth (µm)")
ax.set_ylabel("Ion Count")
ax.set_title("Depth Distribution of Implanted Ions")
ax.legend(loc='upper left')

fig.tight_layout()
plt.show()


dEdx_Be = 828  # stopping power for Be at mean depth (units: e.g. keV/µm)
dEdx_B = 1288   # stopping power for B at mean depth

N_Be = np.sum(df_SRIM_depth_Be10['Be Ions'])  # total Be ions
N_B = np.sum(df_SRIM_depth_B10['B Ions'])     # total B ions

uncertainty_Be = (dEdx_Be * std_Be10) / np.sqrt(N_Be)
uncertainty_B = (dEdx_B * std_B10) / np.sqrt(N_B)

print(f"Be uncertainty = {uncertainty_Be:.4f}")
print(f"B uncertainty = {uncertainty_B:.4f}")