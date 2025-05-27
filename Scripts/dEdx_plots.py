import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def process_srim_stopping_table(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for idx, line in enumerate(lines):
        if "Ion" in line and "Lateral" in line:
            header_idx = idx
            break
    
    data_lines = lines[header_idx + 3:]
    data = []


    for line in data_lines:
        if '-----------' in line or not line.strip():
            break
        parts = line.split()
        if len(parts) >= 4:
            Ion_Energy_str, Ion_Energy_unit, dEdx_elec_str, dEdx_nuc_str, projected_range_str, projected_range_unit = parts[:6]

            # Clean and convert
            Ion_Energy_unit = Ion_Energy_unit.strip("'")
            Ion_Energy = float(Ion_Energy_str.replace(',', '.'))

            projected_range_unit = projected_range_unit.strip("'")
            projected_range = float(projected_range_str.replace(',','.'))

            # Normalize energy to keV
            if Ion_Energy_unit == "eV":
                Ion_Energy /= 1000
            elif Ion_Energy_unit == "MeV":
                Ion_Energy *= 1000

            if projected_range_unit == "A":
                projected_range/= 1e4

            dEdx_elec = float(dEdx_elec_str.replace(',', '.'))
            dEdx_nuc = float(dEdx_nuc_str.replace(',', '.'))

            data.append([Ion_Energy, dEdx_elec, dEdx_nuc, projected_range])
        

    df = pd.DataFrame(data, columns = ['Ion Energy', 'dE/dx elec.', 'dE/dx Nuclear', 'Projected Range'])
    return df         



file_path = r"C:\Users\benja\Desktop\Speciale\Nydata\dE_dx\Beryllium in  H- C (gas).txt"
file_path2 = r"C:\Users\benja\Desktop\Speciale\Nydata\dE_dx\Boron in  H- C (gas).txt"
df = process_srim_stopping_table(file_path)
df2 = process_srim_stopping_table(file_path2)



mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 4))

# Enable grid and customize ticks
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(direction="in", length=6, which="major")
ax.tick_params(direction="in", length=3, which="minor")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.minorticks_on()

# Plot the data
ax.plot(df2["dE/dx elec."]+df2['dE/dx Nuclear'], df2["Projected Range"], color='navy',label = r'$^{10}B$')
#ax.plot(df2["dE/dx elec."], df2["Projected Range"], label = r'$^{10}B$')
#ax.plot(df2['dE/dx Nuclear'], df2["Projected Range"], label = r'$^{10}B$')

ax.plot(df["dE/dx elec."]+df['dE/dx Nuclear'], df["Projected Range"], color='crimson',label = r'$^{10}Be$')
#ax.plot(df["dE/dx elec."], df["Projected Range"], label = r'$^{10}Be$')
#ax.plot(df['dE/dx Nuclear'], df["Projected Range"], label = r'$^{10}Be$')

#ax.plot(df2["Ion Energy"], df2["Projected Range"], label=r'$^{10}B$')
#ax.plot(df["Ion Energy"], df["Projected Range"], label=r'$^{10}Be$')

# Label axes
ax.set_xlabel(r"Total Stopping Power (dE/dx) [$\mathrm{eV}/\langle \mathrm{\AA} \rangle$]")
#ax.set_xlabel(r"Ion energy [KeV]")
ax.set_ylabel(r"Projected Average Range $[\langle \mathrm{\mu m} \rangle]$")

# Show legend
ax.legend()
fig.tight_layout()

# Display plot
billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
plt.savefig(f'{billeder_path}\\dE_dx_plot_for_stoping_power_per_average_range.pdf')
#plt.savefig(f'{billeder_path}\\Ion_energy_per_avg_AA')
plt.show()

E_i = 1398.6 #kev

def exit_energy(initial_energy, data, max_range):
    E_i = initial_energy

    df = data.copy()
    df["dE/dx"] = df["dE/dx elec."] + df["dE/dx Nuclear"]

    #getting E(0)
    E_0 = df["dE/dx"][0]*df["Projected Range"][0]
    
    #getting E(x)

    mask = df["Projected Range"] <= max_range

    lower_df = df[df["Projected Range"] <= max_range].iloc[-1]
    upper_df = df[df["Projected Range"] > max_range].iloc[0]

    x0 = lower_df["Projected Range"]
    x1 = upper_df["Projected Range"]
    y0 = lower_df["dE/dx"]
    y1 = upper_df["dE/dx"]

    # Linear interpolation for dE/dx at max_range
    dEdx_interp = y0 + (y1 - y0) * (max_range - x0) / (x1 - x0)

    E_x = dEdx_interp*max_range

    return E_i-(E_x-E_0)/1000 * 10000#this is in ev/Å * mu m so we need to divide with 1000 to go from ev ->KeV and and multiply with 10.000 to go from å-> mu m

print(exit_energy(E_i, df,1.5150003300000001))

from scipy.integrate import cumulative_trapezoid

def exit_energy_fixed(initial_energy_keV, data, max_range_microns):
    df = data.copy()
    df = df[df["Projected Range"] <= max_range_microns]
    
    if df.empty:
        return initial_energy_keV  # No data within range, no loss assumed
    
    df["dE/dx_total"] = df["dE/dx elec."] + df["dE/dx Nuclear"]  # eV/Å
    df["dE/dx_total"] *= 1e4  # Convert to eV/μm
    
    # Integrate dE/dx over range (μm)
    energy_loss_eV = cumulative_trapezoid(df["dE/dx_total"], df["Projected Range"], initial=0)[-1]  # in eV
    energy_loss_keV = energy_loss_eV / 1000

    remaining_energy_keV = initial_energy_keV - energy_loss_keV

    return max(remaining_energy_keV, 0)
print(exit_energy_fixed(1398.6, df, 1.55))
print(exit_energy_fixed(1398.6, df2, 1.55))


from scipy.integrate import cumulative_trapezoid

def compute_energy_vs_depth(initial_energy_keV, data):
    df = data.copy()
    df["dE/dx_total"] = (df["dE/dx elec."] + df["dE/dx Nuclear"]) * 1e4  # eV/μm

    # Numerical integration of stopping power over depth
    energy_loss_eV = cumulative_trapezoid(df["dE/dx_total"], df["Projected Range"], initial=0)
    energy_loss_keV = energy_loss_eV / 1000

    # Exit energy at each depth
    exit_energy = initial_energy_keV - energy_loss_keV
    exit_energy = np.clip(exit_energy, 0, None)  # Prevent negative energy

    return df["Projected Range"], exit_energy

# Use the function for Boron or Beryllium
depthsbe, energiesbe = compute_energy_vs_depth(E_i, df)  # df2 = Boron
depthsb, energiesb = compute_energy_vs_depth(E_i, df2)  # df2 = Boron

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(direction="in", length=6, which="major")
ax.tick_params(direction="in", length=3, which="minor")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.minorticks_on()
ax.plot(depthsbe, energiesbe, label=r'Exit Energy of $^{10}Be$', color='darkred')
ax.plot(depthsb, energiesb, label=r'Exit Energy of $^{10}B$', color='darkblue')
ax.set_xlabel(r"Depth [$\mu$m]")
ax.set_ylabel(r"Exit Energy [keV]")
ax.set_title("Ion Energy vs. Penetration Depth")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
# plt.savefig("exit_energy_vs_depth.pdf")
plt.savefig(f'{billeder_path}\\exit_energy_vs_depth.pdf')
plt.show()



from numpy import gradient

# Prepare data for Boron (df2)
df2["dE/dx_total"] = (df2["dE/dx elec."] + df2["dE/dx Nuclear"]) * 1e4  # eV/μm
depths_b, energies_b = compute_energy_vs_depth(E_i, df2)
dE_dx_num_b = -gradient(energies_b, depths_b) * 1000  # eV/μm

# Prepare data for Beryllium (df)
df["dE/dx_total"] = (df["dE/dx elec."] + df["dE/dx Nuclear"]) * 1e4  # eV/μm
depths_be, energies_be = compute_energy_vs_depth(E_i, df)
dE_dx_num_be = -gradient(energies_be, depths_be) * 1000  # eV/μm

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]}, 
                               figsize=(10, 7))

# Top plot: SRIM vs Numerical dE/dx for B and Be
ax1.plot(df2["Projected Range"], df2["dE/dx_total"]/1e3, label=r"SRIM dE/dx $^{10}B$", color="darkblue")
ax1.plot(depths_b, dE_dx_num_b/1e3, '--', label=r"Numerical dE/dx $^{10}B$", color="green")

ax1.plot(df["Projected Range"], df["dE/dx_total"]/1e3, label=r"SRIM dE/dx $^{10}Be$", color="darkred")
ax1.plot(depths_be, dE_dx_num_be/1e3, '--', label=r"Numerical dE/dx $^{10}Be$", color="orange")

ax1.set_ylabel(r"Total Stopping Power [keV/$\mu$m]")
ax1.set_title("Comparison of SRIM and Numerical dE/dx")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.tick_params(direction="in", length=6, which="major")
ax1.tick_params(direction="in", length=3, which="minor")
ax1.xaxis.set_ticks_position("both")
ax1.yaxis.set_ticks_position("both")
ax1.minorticks_on()

# Bottom plot: Residuals (SRIM - Numerical) for B and Be
residuals_b = (df2["dE/dx_total"]/1e3) - (dE_dx_num_b/1e3)
residuals_be = (df["dE/dx_total"]/1e3) - (dE_dx_num_be/1e3)

ax2.plot(df2["Projected Range"], residuals_b, label=r"Residuals $^{10}B$", color="blue")
ax2.plot(df["Projected Range"], residuals_be, label=r"Residuals $^{10}Be$", color="red")

ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_xlabel(r"Depth [$\mu$m]")
ax2.set_ylabel("Residuals\n[keV/$\mu$m]")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
ax2.tick_params(direction="in", length=6, which="major")
ax2.tick_params(direction="in", length=3, which="minor")
ax2.xaxis.set_ticks_position("both")
ax2.yaxis.set_ticks_position("both")
ax2.minorticks_on()

plt.tight_layout()
plt.savefig(f'{billeder_path}\\qualityofanalyticalmethod.pdf')

plt.show()
