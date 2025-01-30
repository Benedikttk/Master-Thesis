import pandas as pd

def read_exyz_file(file_path):
    # Define correct headers manually
    column_names = [
        "Ion Number", "Energy (keV)", "Depth (X) (Angstrom)", 
        "Y (Angstrom)", "Z (Angstrom)", "Electronic Stop.(eV/A)", 
        "Energy lost due to Last Recoil(eV)"
    ]

    # Read the file, skipping metadata and header separator line
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=15, names=column_names)

    # Convert numeric columns (replacing commas with dots for floats)
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

    return df


