import numpy as np

def calculate_fractions(cutoff_depth, df_srim_depth_Be10, df_srim_depth_B10):
    total_ions_Be10 = df_srim_depth_Be10['Be Ions'].sum()
    total_ions_B10 = df_srim_depth_B10['B Ions'].sum()

    Be10_after_cutoff = df_srim_depth_Be10[df_srim_depth_Be10['Depth (Angstrom)'] > cutoff_depth]['Be Ions'].sum()
    B10_before_cutoff = df_srim_depth_B10[df_srim_depth_B10['Depth (Angstrom)'] <= cutoff_depth]['B Ions'].sum()

    Be10_fraction = Be10_after_cutoff / total_ions_Be10
    B10_fraction = B10_before_cutoff / total_ions_B10

    return Be10_fraction, B10_fraction

def calculate_delta_e_e_res(df, anode_1_length, anode_2_length, effective_length):
    delta_e_values = []
    e_res_values = []

    for ion_number in range(1, max(df["Ion Number"]) + 1):
        ion_data = df[df["Ion Number"] == ion_number]
        x_positions = ion_data["Depth (X) (Angstrom)"]
        energies = ion_data["Energy (keV)"]

        if (x_positions.max() >= anode_2_length and
            x_positions.max() <= effective_length and
            x_positions.min() <= anode_1_length):

            delta_e = energies.iloc[0] - energies.iloc[np.where(x_positions >= anode_1_length)[0][0]]
            e_res_index = np.where(x_positions >= anode_2_length)[0][0]
            e_res = energies.iloc[e_res_index]

            if e_res > 0:
                delta_e_values.append(delta_e)
                e_res_values.append(e_res)

    return delta_e_values, e_res_values