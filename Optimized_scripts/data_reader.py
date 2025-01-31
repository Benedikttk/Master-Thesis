import pandas as pd

def read_srim_file(file_path, ion_type):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for idx, line in enumerate(lines):
        if line.strip().startswith("DEPTH"):
            header_index = idx
            break

    data_lines = lines[header_index + 1:]
    data = []

    for line in data_lines:
        if line.strip():
            try:
                depth, ions, _ = line.split()
                data.append([float(depth.replace(',', '.')), float(ions.replace(',', '.'))])
            except ValueError:
                continue

    df = pd.DataFrame(data, columns=['Depth (Angstrom)', f'{ion_type} Ions'])
    return df

def read_exyz_file(file_path):
    column_names = [
        "Ion Number", "Energy (keV)", "Depth (X) (Angstrom)", 
        "Y (Angstrom)", "Z (Angstrom)", "Electronic Stop.(eV/A)", 
        "Energy lost due to Last Recoil(eV)"
    ]
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=15, names=column_names)
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
    return df