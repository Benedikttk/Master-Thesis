import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

