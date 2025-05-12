import pandas as pd

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



file_path = r"C:\Users\benja\Desktop\Speciale\Ny data\dE_dx\BerylliuminSi-N.txt"
file_path2 = r"C:\Users\benja\Desktop\Speciale\Ny data\dE_dx\BoroninSi-N.txt"
df = process_srim_stopping_table(file_path)
df2 = process_srim_stopping_table(file_path2)





import matplotlib.pyplot as plt
plt.plot(df2["dE/dx elec."]+df2['dE/dx Nuclear'], df2["Projected Range"], label = 'B')

plt.plot(df["dE/dx elec."]+df['dE/dx Nuclear'], df["Projected Range"], label = 'Be')

#plt.plot(df2["Ion Energy"], df2["Projected Range"], label = 'B')

#plt.plot(df["Ion Energy"], df["Projected Range"], label = 'Be')

plt.xlabel("dE/dx")
plt.ylabel("Projected Range")

plt.legend()
plt.show()