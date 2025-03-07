import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Functions import read_exyz_file


# Create a plot
fig, axs = plt.subplots()

img = mpimg.imread(r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder\DetectorFigure.png')

axs.imshow(img, extent=[-190,410,-300,300]) #[x højre, , y ned, y op]

#trajectories of ions
file_path = r"C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\EXYZs\EXYZ1.txt"  
df = read_exyz_file(file_path)
print(df)

# Convert Depth X and Y from Angstroms to millimeters
df['Depth (X) (mm)'] = df['Depth (X) (Angstrom)'] * 1e-7
df['Y (mm)'] = df['Y (Angstrom)'] * 1e-7


# Loop through all unique ion numbers
for ion_number in df["Ion Number"].unique():
    # Filter data for the current ion number
    filtered_df = df[df["Ion Number"] == ion_number]

    # Plot Depth X (in mm) vs Depth Y (in mm) for the current ion number
    axs.plot(filtered_df['Depth (X) (mm)'], filtered_df['Y (mm)'], label=f'Ion {ion_number}')
    print(max(filtered_df['Depth (X) (mm)']))


# Labels and title
plt.xlabel('Depth X (mm)')
plt.ylabel('Depth Y (mm)')
plt.tight_layout()
plt.axis('off')

billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'

plt.savefig(f'{billeder_path}\\SRIM_Detector_SIMS.pdf')
plt.show()