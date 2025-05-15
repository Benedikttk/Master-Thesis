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



file_path = r"C:\Users\benja\Desktop\Speciale\Nydata\dE_dx\BerylliuminSi-N.txt"
file_path2 = r"C:\Users\benja\Desktop\Speciale\Nydata\dE_dx\BoroninSi-N.txt"
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
fig, ax = plt.subplots()

# Enable grid and customize ticks
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(direction="in", length=6, which="major")
ax.tick_params(direction="in", length=3, which="minor")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.minorticks_on()

# Plot the data
ax.plot(df2["dE/dx elec."]+df2['dE/dx Nuclear'], df2["Projected Range"], label = r'$^{10}B$')
#ax.plot(df2["dE/dx elec."], df2["Projected Range"], label = r'$^{10}B$')
#ax.plot(df2['dE/dx Nuclear'], df2["Projected Range"], label = r'$^{10}B$')

ax.plot(df["dE/dx elec."]+df['dE/dx Nuclear'], df["Projected Range"], label = r'$^{10}Be$')
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

# Display plot
#billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'
#plt.savefig(f'{billeder_path}\\dE_dx_plot_for_stoping_power_per_average_range.pdf')
#plt.savefig(f'{billeder_path}\\Ion_energy_per_avg_AA')
#plt.show()

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
print(exit_energy_fixed(1398.6, df, 1.5150003300000001))
print(exit_energy_fixed(1398.6, df2, 1.5150003300000001))


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
depths, energies = compute_energy_vs_depth(E_i, df2)  # df2 = Boron

# Plot
fig, ax = plt.subplots()
ax.plot(depths, energies, label=r'Exit Energy of $^{10}B$', color='darkred')
ax.set_xlabel(r"Depth [$\mu$m]")
ax.set_ylabel(r"Exit Energy [keV]")
ax.set_title("Ion Energy vs. Penetration Depth")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
# plt.savefig("exit_energy_vs_depth.pdf")
plt.show()



from numpy import gradient

# 1. Get dE/dx from SRIM
df2["dE/dx_total"] = (df2["dE/dx elec."] + df2["dE/dx Nuclear"]) * 1e4  # eV/μm

# 2. Compute energy vs. depth
depths, energies = compute_energy_vs_depth(E_i, df2)

# 3. Compute -dE/dx from the energy-depth curve
# Since E(x) decreases, dE/dx = -d(E)/dx
dE_dx_from_energy = -gradient(energies, depths) * 1000  # convert keV/μm to eV/μm for matching units

# 4. Plot both
fig, ax = plt.subplots()
ax.plot(df2["Projected Range"], df2["dE/dx_total"], label="SRIM: dE/dx", color="blue")
ax.plot(depths, dE_dx_from_energy, '--', label="Numerical dE/dx (from E(x))", color="orange")

ax.set_xlabel(r"Depth [$\mu$m]")
ax.set_ylabel(r"Total Stopping Power [eV/$\mu$m]")
ax.set_title("Comparison of SRIM dE/dx and Numerical dE/dx")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
