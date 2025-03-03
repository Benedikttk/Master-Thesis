import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Functions import FileCheck, deltE_Efinal

# 
Filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
Subject = "[CDAT0"

files = [i for i in FileCheck(filepath=Filepath, endswith=".txt.mpa")]
data = deltE_Efinal(filepath=Filepath, subject=Subject, filename=files[0]) 

#-------PLOTTTTINGGG-----
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.12, wspace=0.12)

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
#plt.savefig(f'{billeder_path}\\deltaE_Efinal.pdf')

plt.tight_layout()
plt.show()

