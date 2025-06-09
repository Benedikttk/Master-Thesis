import matplotlib.pyplot as plt
import numpy as np

# Foil thickness (µm)
thicknesses = np.array([1.3, 1.4, 1.51, 1.55, 1.6])

# Pressure window bounds (mbar)
p_mins = np.array([10.89, 9.12, 6.75, 6.11, 6.11])
p_maxs = np.array([19.87, 18.36, 17.65, 16.67, 16.67])

# Calculate average pressure and error bars
p_avgs = (p_mins + p_maxs) / 2
p_err_lower = p_avgs - p_mins
p_err_upper = p_maxs - p_avgs
p_err = [p_err_lower, p_err_upper]

fig, ax = plt.subplots(figsize=(10, 4))

ax.grid(True, linestyle='--', alpha=0.6)

# Set tick params on the ax object
ax.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax.minorticks_on()

ax.errorbar(
    thicknesses,
    p_avgs,
    yerr=p_err,
    xerr=thicknesses * 0.01,
    fmt='o',
    capsize=5,
    capthick=2,
    markerfacecolor='darkred',
    markersize=8,
    linestyle='',
    color='darkred',
    label='Average Operating Pressure ± Uncertainty'
)

print(p_avgs, p_err)
ax.set_xlabel(r'$Si_{3}N_{4}$ Thickness (µm)', fontsize=14)
ax.set_ylabel(r'$C_{4}H_{10}$ Pressure (mbar)', fontsize=14)
ax.set_title('GIC pressure proposal', fontsize=16)
ax.set_xticks(thicknesses)
ax.set_xticklabels([f'{t:.2f}' for t in thicknesses])

ax.legend()
fig.tight_layout()

billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
fig.savefig(f'{billeder_path}\\pressurewindow.pdf')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
enthickness = np.array([
    13994.12, 14500.00, 14981.25, 15050.00, 15547.06, 15900.00, 16443.75, 16800.00, 17333.33,
    16450, 15560, 16390, 16000, 16179, 16700, 16500, 16500, 16800, 16800, 17333.33
])
thickness = enthickness / 1e4  # Convert to micrometers
energy = np.array([
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
    1250, 1260, 1275, 1290, 1301, 1316, 1330, 1344, 1357, 1371, 1385
])

# Constants (amu)
Mp = 10           # mass of 10Be+
Mi = 10 + 16  # mass of 10Be16O-

q_prime = 1             # charge state after stripping
Vi = 0                  # initial energy of injected anions (keV)

# Calculate terminal voltage Vt (in kV)
# Rearranged from:
# Ef = ((Vi + Vt)*Mp / Mi) + q'*Vt
# => Vt = (Ef - (Vi*Mp/Mi)) / (Mp/Mi + q')

Vt = (energy - (Vi * Mp / Mi)) / (Mp / Mi + q_prime)  # units consistent with energy in keV
# Vt will be in kV since energy is in keV and charge in units of e

# Define linear model for fitting
def linmodel(x, a, b):
    return a * x + b

# Fit thickness vs energy
popt, pcov = curve_fit(linmodel, energy, thickness)
a, b = popt
a_err, b_err = np.sqrt(np.diag(pcov))

# Prepare for plotting
x_fit = np.linspace(min(energy), max(energy), 200)
y_fit = linmodel(x_fit, a, b)

fig, ax1 = plt.subplots(figsize=(10, 5))

# Scatter and fit line on bottom x-axis (Energy)
ax1.scatter(energy, thickness, color='blue', label='TRIM Data')
ax1.plot(x_fit, y_fit, color='red',
         label=fr'Fit: $y = ax + b$')

ax1.set_xlabel('Simulated Energy (keV)', fontsize=14)
ax1.set_ylabel(r'Optimal Cutoff Depth ($\mu$m)', fontsize=14)
ax1.tick_params(axis='x', direction='in', length=6, which='major')
ax1.tick_params(axis='x', direction='in', length=3, which='minor')
ax1.minorticks_on()
ax1.grid(True, linestyle='--', alpha=0.6)

# Create twin axis for Terminal Voltage on top
ax2 = ax1.twiny()

# Set top x-axis limits same as bottom
ax2.set_xlim(ax1.get_xlim())

# Select ticks for terminal voltage axis (choose nice values within Vt range)
vt_ticks = np.linspace(min(Vt), max(Vt),9)
# Convert vt_ticks back to energy for positioning on bottom axis
energy_ticks = vt_ticks * (Mp / Mi + q_prime) + (Vi * Mp / Mi)

ax2.set_xticks(energy_ticks)
ax2.set_xticklabels([f'{v:.0f}' for v in vt_ticks])
ax2.set_xlabel('Terminal Voltage (kV)', fontsize=14)

ax2.tick_params(axis='x', direction='in', length=6, which='major')
ax2.tick_params(axis='x', direction='in', length=3, which='minor')
ax2.minorticks_on()

ax1.legend()
plt.tight_layout()

# Optional: Save figure
billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
fig.savefig(f'{billeder_path}\\OptimalCutoffDepthVsEnergy.pdf')

plt.show()

# Print fit results
print(f"a = {a:.5f} ± {a_err:.5f} μm/keV")
print(f"b = {b:.5f} ± {b_err:.5f} μm")

