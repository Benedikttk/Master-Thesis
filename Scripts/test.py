from ReadingFromSRIMfile import process_file
import numpy as np  
import os

file_path_Be10 = r"C:\Users\benja\Desktop\Speciale\Scripts\RANGE_1400_ion_1000Be10.txt"

# Check if the file exists
if os.path.isfile(file_path_Be10):
    print(f"The file at {file_path_Be10} exists.")
else:
    print(f"The file at {file_path_Be10} does not exist.")


