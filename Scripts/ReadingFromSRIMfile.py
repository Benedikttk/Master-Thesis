import pandas as pd

# Function to process a file and return the DataFrame
def process_file(file_path, ion_type):
    # Open and process the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Locate the header line
    for idx, line in enumerate(lines):
        if line.strip().startswith("DEPTH"):
            header_index = idx
            break

    # Extract data lines after the header
    data_lines = lines[header_index + 1:]
    data = []

    for line in data_lines:
        if line.strip():  # Skip empty lines
            try:
                # Split columns and replace ',' with '.' for numeric parsing
                depth, ions, _ = line.split()
                data.append([float(depth.replace(',', '.')), float(ions.replace(',', '.'))])
            except ValueError:
                continue  # Ignore malformed lines

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Depth (Angstrom)', f'{ion_type} Ions'])
    return df
    

