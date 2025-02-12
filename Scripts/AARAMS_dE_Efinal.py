import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'


subject = "[CDAT0"

for file in os.listdir(filepath):
    if file.startswith("Be"):
        print(file)
    else:
        print("The other files: {}".format(file))
finished = True
print("\n")

list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(".txt.mpa")]
print("These are the raw files", list_of_raw_files)

with open(filepath + "\\" + list_of_raw_files[0], 'r') as file:
    lines = file.readlines()
    if lines ==[]:
        print("The file is empty")
    else:
        print("The file is not empty")

header_index = None
for idx, line in enumerate(lines):
    if subject in line:
        header_index = idx
        print(f"Printing the header index of {subject}:", header_index)
        break
if header_index is not None:
    print("The line corresponding to header_index is:", lines[header_index])
else:
    print("No matching header found.")
# Now we have the index of the line we want to start reading from, this is also where the dE/E_final data starts.
# Now we want to read the data from the file and put it into a dataframe.

print(type(lines))

data_index = None
new_lines = lines[header_index:]
for idx, line in enumerate(new_lines):
    if line.startswith("[DATA]"):
        data_index = idx
        print(f"Printing the data index of [DATA]:", data_index)
        break

data_start = data_index + 1

#good way forthis data sheet as after every data there is text so it stops at the first text
data = []
for line in new_lines[data_start:]:
    if any(char.isalpha() for char in line):
        break
    data.append(line.strip().split())

print("checking last element in list",data[-1])
print("checking first element in list",data[0])

# Now we want to put the data into a dataframe
data = pd.DataFrame(data, columns=["E_final", "dE", "counts"])
data["dE"] = pd.to_numeric(data["dE"])
data["E_final"] = pd.to_numeric(data["E_final"])
data["counts"] = pd.to_numeric(data["counts"])
print(data.head())

'''plt.scatter(data["E_final"], data["dE"], c=data["counts"], cmap='viridis', s=1)  # Adjust the 's' parameter for dot size
plt.xlabel("E_final [keV]")
plt.ylabel("dE [keV]")
plt.colorbar()
print(data["counts"])

plt.scatter(data["dE"], data["counts"], s=1)
plt.scatter(data["E_final"], data["counts"], s=1)
'''
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

# Main scatter plot (center)
ax_main = fig.add_subplot(gs[1, 0])
sc = ax_main.scatter(data["E_final"], data["dE"], c=data["counts"], cmap='viridis', s=2)
ax_main.set_ylabel("dE [keV]")
ax_main.set_xlabel(r"$E_{final} [keV]$")

ax_main.grid(True, linestyle='--', alpha=0.6)
ax_main.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax_main.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax_main.xaxis.set_ticks_position("both")
ax_main.yaxis.set_ticks_position("both")
ax_main.minorticks_on()

# Top marginal plot (E_final distribution)
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
sns.kdeplot(data["E_final"], fill=True, color="black", ax=ax_top)

ax_top.grid(True, linestyle='--', alpha=0.6)
ax_top.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax_top.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
#ax_top.set_xticklabels([])
ax_top.xaxis.set_ticks_position("both")
ax_top.yaxis.set_ticks_position("both")
ax_top.minorticks_on()
ax_top.set_xlabel(None)
ax_top.set_ylabel(r"$\rho(dE)$")

# Right marginal plot (dE distribution)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
sns.kdeplot(data["dE"], fill=True, color="black", ax=ax_right, vertical=True)

ax_right.grid(True, linestyle='--', alpha=0.6)
ax_right.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax_right.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
#ax_right.set_yticklabels([])
ax_right.xaxis.set_ticks_position("both")
ax_right.yaxis.set_ticks_position("both")
ax_right.minorticks_on()
ax_right.set_xlabel(r"$\rho(E_{final})$")
ax_right.set_ylabel(None)



# Colorbar
cbar = plt.colorbar(sc, ax=ax_right, orientation="vertical", fraction=0.5)
cbar.set_label("Counts")


# Ensure all ticks (major and minor) point inward with different lengths
for ax in [ax_main, ax_top, ax_right]:
    ax.tick_params(direction="in", length=6, which="major")  # Longer major ticks
    ax.tick_params(direction="in", length=3, which="minor")  # Shorter minor ticks

billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'
plt.savefig(f'{billeder_path}\\deltaE_Efinal.pdf')

plt.tight_layout()
plt.show()

