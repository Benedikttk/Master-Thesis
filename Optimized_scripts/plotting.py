import matplotlib.pyplot as plt

def plot_depth_distribution(df_srim_depth_Be10, df_srim_depth_B10, cutoff_depth, best_cutoff_depth, save_path):
    # Function implementation
    plt.figure(figsize=(10, 6))
    plt.hist(df_srim_depth_Be10['Depth (Angstrom)'], bins=len(df_srim_depth_Be10['Depth (Angstrom)']),
             weights=df_srim_depth_Be10['Be Ions'], color='red', alpha=0.6, histtype='stepfilled', label='Be10')
    plt.hist(df_srim_depth_B10['Depth (Angstrom)'], bins=len(df_srim_depth_B10['Depth (Angstrom)']),
             weights=df_srim_depth_B10['B Ions'], color='blue', alpha=0.4, histtype='stepfilled', label='B10')
    plt.axvline(x=cutoff_depth, color='black', linestyle='--', label='Cutoff for B10')
    plt.axvline(x=best_cutoff_depth, color='green', linestyle='--', label='Optimized Cutoff')
    plt.xlabel('Depth (Angstrom)')
    plt.ylabel('Ion Count')
    plt.title('Depth Distribution of Be10 and B10 Ions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}\\BoronSupressionDepth.pdf')
    plt.show()

def plot_delta_e_e_res(boron_e_res, boron_delta_e, beryllium_e_res, beryllium_delta_e, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(boron_e_res, boron_delta_e, alpha=0.5, label="Boron", color="blue")
    plt.scatter(beryllium_e_res, beryllium_delta_e, alpha=0.5, label="Beryllium", color="red")
    plt.xlabel('Residual Energy (E_res) [keV]')
    plt.ylabel('Energy Loss (Delta E) [keV]')
    plt.title('Delta E vs E_res for Boron and Beryllium')
    plt.legend()
    plt.savefig(f'{save_path}\\deltaE_Eres.pdf')
    plt.show()