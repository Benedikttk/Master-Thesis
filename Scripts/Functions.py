import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Mostely for AArAMS_dE_Efinal.py
def FileCheck(filepath, endswith=None):
    """
    Check files in the given directory and return a list of files ending with the specified condition.

    Parameters:
    filepath (str): The path to the directory to check.
    endswith (str): The file extension to filter files.

    Returns:
    list: A list of filenames ending with the specified condition.
    """
    if endswith:
        list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(endswith)]
        return list_of_raw_files
    else:
        for file in os.listdir(filepath):
            if file.startswith("Be"):
                print(file)
            else:
                print(f"The other files: {file}")
        list_of_files = [file for file in os.listdir(filepath)]
        return list_of_files

def deltE_Efinal(filepath, subject, filename):
    """
    Extract values from AARAMS dE/Efinal plots.

    Parameters:
    filepath (str): The path to the directory containing the file.
    subject (str): The subject string to search for in the file.
    filename (str): The name of the file to process.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted data.
    """
    if not filename.endswith(".txt.mpa"):
        print("Invalid file type. Please provide a .txt.mpa file.")
        return

    full_path = os.path.join(filepath, filename)
    if not os.path.exists(full_path):
        print(f"File {filename} does not exist in the specified directory.")
        return

    with open(full_path, 'r') as file:
        lines = file.readlines()
        if not lines:
            print("The file is empty")
            return
        else:
            print("The file is not empty")

    header_index = next((idx for idx, line in enumerate(lines) if subject in line), None)
    if header_index is None:
        print("No matching header found.")
        return
    print(f"Printing the header index of {subject}: {header_index}")
    print(f"The line corresponding to header_index is: {lines[header_index]}")

    new_lines = lines[header_index:]
    data_index = next((idx for idx, line in enumerate(new_lines) if line.startswith("[DATA]")), None)
    if data_index is None:
        print("No data section found.")
        return
    print(f"Printing the data index of [DATA]: {data_index}")

    data_start = data_index + 1

    data = []
    for line in new_lines[data_start:]:
        if any(char.isalpha() for char in line):
            break
        data.append(line.strip().split())

    data_df = pd.DataFrame(data, columns=["E_final", "dE", "counts"])
    data_df = data_df.apply(pd.to_numeric)
    return data_df

#This is for the CalcPenDepthSRIM.py file extraction
# Function to process a file and return the DataFrame
def process_file(file_path, ion_type):
    """
    Processes a given file to extract and return depth and ion data.

    Parameters:
    file_path (str): The path to the file to be processed.
    ion_type (str): The type of ion (used in the DataFrame column name).

    Returns:
    pd.DataFrame: A DataFrame containing depth and ion data.
    """
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
    

# Function to calculate the fraction of Be10 and B10 ions based on a given cutoff depth
def calculate_fractions(cutoff_depth, df_SRIM_depth_Be10, df_SRIM_depth_B10, total_ions_Be10, total_ions_B10):
    """
    Calculates the fraction of Be10 and B10 ions based on a specified cutoff depth.

    Parameters:
    cutoff_depth (float): The depth at which the ions are separated.
    df_SRIM_depth_Be10 (pd.DataFrame): DataFrame containing the depth and Be10 ion data.
    df_SRIM_depth_B10 (pd.DataFrame): DataFrame containing the depth and B10 ion data.
    total_ions_Be10 (float): Total number of Be10 ions.
    total_ions_B10 (float): Total number of B10 ions.

    Returns:
    tuple: The fraction of Be10 ions after the cutoff depth and the fraction of B10 ions before the cutoff depth.
    """
    Be10_after_cutoff = df_SRIM_depth_Be10[df_SRIM_depth_Be10['Depth (Angstrom)'] > cutoff_depth]['Be Ions'].sum()
    B10_before_cutoff = df_SRIM_depth_B10[df_SRIM_depth_B10['Depth (Angstrom)'] <= cutoff_depth]['B Ions'].sum()
    
    Be10_fraction = Be10_after_cutoff / total_ions_Be10
    B10_fraction = B10_before_cutoff / total_ions_B10
    
    return Be10_fraction, B10_fraction

    

#For ReadingEfficiencyfromAARAMS
def get_txt_files(directory: str, extension: str = ".txt") -> list:
    """
    Returns a list of files with the specified extension in the given directory.

    Parameters:
    directory (str): The directory to search for the files.
    extension (str): The file extension to look for. Default is ".txt".

    Returns:
    list: A list of filenames ending with the specified extension.
    """
    return [file for file in os.listdir(directory) if file.endswith(extension)]

def read_block_data(filepath: str) -> list:
    """
    Reads a file and extracts data from the [BLOCK DATA] section.

    Parameters:
    filepath (str): The path to the file to be read.

    Returns:
    list: A list of lines containing the block data, or an empty list if no data found.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    if not lines:
        print("The file is empty")
        return []
    
    for idx, line in enumerate(lines):
        if line.strip().startswith("[BLOCK DATA]"):
            return lines[idx + 1:]
    
    return []

def parse_dataframe(data_lines: list, column_names: list) -> pd.DataFrame:
    """
    Converts raw data lines into a cleaned Pandas DataFrame.

    Parameters:
    data_lines (list): The raw lines of data to convert.
    column_names (list): A list of column names for the DataFrame.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.DataFrame([line.split() for line in data_lines], columns=column_names)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.iloc[1:]  # Drop first row (potential header duplication)

def calculate_Be10_statistics(df: pd.DataFrame) -> tuple:
    """
    Computes mean and standard deviation of Be10 counts.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the Be10 data.

    Returns:
    tuple: The mean and standard deviation of Be10 counts.
    """
    Be10cnts = df["10Becnts"].astype(float)
    return Be10cnts.mean(), Be10cnts.std()

def calculate_Be10_current(avg_Be10cnts: float, avg_time: float, time_uncertainty: float) -> tuple:
    """
    Computes the current of Be10 and its uncertainty.

    Parameters:
    avg_Be10cnts (float): The average number of Be10 counts.
    avg_time (float): The average time in seconds.
    time_uncertainty (float): The uncertainty in the time.

    Returns:
    tuple: The Be10 current and its uncertainty.
    """
    Q_Be10 = 1.6e-19 * avg_Be10cnts
    I_Be10 = Q_Be10 / avg_time * 1e6  # Convert to microamperes
    I_Be10_uncertainty = (Q_Be10 / avg_time**2) * time_uncertainty
    return I_Be10, I_Be10_uncertainty

def extract_metadata(filepath: str, key_name: str) -> float:
    """
    Extracts a specific metadata value from a file.

    Parameters:
    filepath (str): The path to the file to be read.
    key_name (str): The key name to search for in the file.

    Returns:
    float: The extracted metadata value, or None if not found or malformed.
    """
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip().startswith(key_name):
                try:
                    return float(line.strip().split(":")[1])
                except ValueError:
                    return None
    return None

def calculate_Be9_ions(Be9_current: float, detector_live_time: float) -> float:
    """
    Computes the number of Be9 ions.

    Parameters:
    Be9_current (float): The Be9 current.
    detector_live_time (float): The detector live time in seconds.

    Returns:
    float: The number of Be9 ions.
    """
    return (Be9_current * detector_live_time) / 1.6e-19

def calculate_ratio_and_efficiency(avg_Be10cnts: float, Be9cnts: float, std_Be10cnts: float, R_nominiel: float, R_nominiel_uncertainty: float, runs) -> tuple:
    """
    Computes the Be10/Be9 ratio and isotropic ratio efficiency with uncertainties.

    Parameters:
    avg_Be10cnts (float): The average Be10 counts.
    Be9cnts (float): The Be9 counts.
    std_Be10cnts (float): The standard deviation of Be10 counts.
    R_nominiel (float): The nominal Be10/Be9 ratio.
    R_nominiel_uncertainty (float): The uncertainty in the nominal Be10/Be9 ratio.

    Returns:
    tuple: The Be10/Be9 ratio, its uncertainty, isotropic ratio efficiency, and its uncertainty.
    """
    sumed_avg_Be10cnts = avg_Be10cnts * runs 
    R_n = sumed_avg_Be10cnts / Be9cnts
    R_n_uncertainty = R_n * np.sqrt((std_Be10cnts / sumed_avg_Be10cnts)**2)
    
    iso_eff = (R_n / R_nominiel) * 100
    iso_eff_uncertainty = iso_eff * np.sqrt((R_n_uncertainty / R_n)**2 + (R_nominiel_uncertainty / R_nominiel)**2)

    return R_n, R_n_uncertainty, iso_eff, iso_eff_uncertainty


#OptimizedGasDensity

# Function to count valid ions and calculate average length
def count_valid_ions(df, effective_length, anode_1_length):
    """
    Count the number of valid ions and calculate the average maximum length and uncertainty (standard deviation) of valid ions.
    
    A valid ion is defined as one that:
    - Has a maximum X-position greater than or equal to the first anode length.
    - Has a maximum X-position less than or equal to the effective length.
    - Has a minimum X-position less than or equal to the first anode length.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing ion data, including 'Ion Number', 'Depth (X) (Angstrom)' columns for ion positions.
    
    effective_length : float
        The maximum effective length (in Angstroms) that an ion can travel.
    
    anode_1_length : float
        The length (in Angstroms) of the first anode.

    Returns:
    --------
    valid_ions : int
        The number of valid ions that meet the above criteria.
    
    avg_length : float
        The average maximum X-position of valid ions (in Angstroms).
    
    uncertainty : float
        The standard deviation of the maximum X-positions of valid ions (in Angstroms).
    """
    valid_ions = 0
    max_lengths = []  # Store max lengths of valid ions
    
    for ion_number in range(1, max(df["Ion Number"]) + 1):
        ion_data = df[df["Ion Number"] == ion_number]
        x_positions = ion_data["Depth (X) (Angstrom)"]
        
        # Check if ion crosses first anode and reaches second anode without surpassing effective length
        if (x_positions.max() >= anode_1_length and 
            x_positions.max() <= effective_length and 
            x_positions.min() <= anode_1_length):
            valid_ions += 1
            max_lengths.append(x_positions.max())  # Store max length of valid ion
    
    # Calculate average length and uncertainty (standard deviation)
    if max_lengths:
        avg_length = np.mean(max_lengths)
        uncertainty = np.std(max_lengths)
    else:
        avg_length = 0
        uncertainty = 0
    
    return valid_ions, avg_length, uncertainty

#EXYZreader moved to here!!!
def read_exyz_file(file_path):
    """
    Reads an EXYZ file and converts it into a Pandas DataFrame with properly named columns.

    The function processes a file containing ion data, skips metadata and irrelevant header lines, 
    and assigns custom column names. It also handles the conversion of numeric values, ensuring that 
    commas in numeric entries are replaced with periods to ensure proper float conversion.

    Parameters:
    -----------
    file_path : str
        The path to the EXYZ file to be read.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the ion data with the following columns:
        - "Ion Number": The ion number.
        - "Energy (keV)": The energy of the ion in keV.
        - "Depth (X) (Angstrom)": The depth of the ion in Angstroms (X position).
        - "Y (Angstrom)": The Y position of the ion in Angstroms.
        - "Z (Angstrom)": The Z position of the ion in Angstroms.
        - "Electronic Stop.(eV/A)": The electronic stopping power (eV/Å).
        - "Energy lost due to Last Recoil(eV)": The energy lost due to the last recoil event (eV).
    
    Notes:
    ------
    - The function assumes that the first 15 lines of the file contain metadata or header information 
      that is to be skipped.
    - Non-numeric values or errors encountered during the conversion of the columns are set to NaN 
      (Not a Number).
    - The function replaces commas with periods in numeric columns to ensure proper float conversion 
      when the file uses commas as decimal separators (common in some regional settings).

    Example:
    --------
    df = read_exyz_file('path/to/exyz_file.csv')
    """
    
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


#KNNAARAMS functions

def extract_data_from_mpa(filepath, subject, file_index=1, info=None):
    """
    Extracts numerical data from a `.txt.mpa` file in a specified directory.

    This function:
    - Lists all `.txt.mpa` files in the given directory.
    - Reads the specified file (default is the second file).
    - Finds the header line that matches the given subject.
    - Identifies the start of the `[DATA]` section.
    - Extracts numerical data and returns it as a pandas DataFrame.

    Parameters:
    filepath (str): The directory containing `.txt.mpa` files.
    subject (str): The subject string used to locate the header within the file.
    file_index (int): The index of the file to read (default is 1, the second file).
    info (str or None): If set to None, no information will be printed.

    Returns:
    pd.DataFrame: A DataFrame containing extracted numerical data with columns ["E_final", "dE", "counts"].

    Raises:
    ValueError: If no `.txt.mpa` files are found, the file is empty, the subject header is missing, 
                or the `[DATA]` section is not found.
    """
    # Get list of matching files
    files = os.listdir(filepath)
    raw_files = [file for file in files if file.endswith(".txt.mpa")] #hardcoded

    if not raw_files:
        raise ValueError("No raw files found.")

    # If info is not None, print the files
    if info is not None:
        print("\n".join(raw_files))

    # Check if the file_index is valid
    if file_index < 1 or file_index > len(raw_files):
        raise ValueError(f"Invalid file_index: {file_index}. The directory contains {len(raw_files)} files.")

    # Read the specified file
    file_path = os.path.join(filepath, raw_files[file_index - 1])

    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        raise ValueError("The file is empty")

    # Find header index
    header_index = next((idx for idx, line in enumerate(lines) if subject in line), None)
    if header_index is None:
        raise ValueError(f"No matching header found for {subject}")

    # Find data index
    data_start = next((header_index + idx + 1 for idx, line in enumerate(lines[header_index:]) if line.startswith("[DATA]")), None)
    if data_start is None:
        raise ValueError("No data section found.")

    # Extract data
    data = []
    for line in lines[data_start:]:
        if any(char.isalpha() for char in line):  # Stop when encountering a line with alphabetic characters
            break
        data.append(line.strip().split())

    # Create DataFrame
    data = pd.DataFrame(data, columns=["E_final", "dE", "counts"]).apply(pd.to_numeric)

    return data

def SilhouetteScore_to_Confidence(SilhouetteScore):
    Confidence = (SilhouetteScore + 1)/2 *100
    return Confidence