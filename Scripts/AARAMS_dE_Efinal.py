import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'


subject = "[CDAT0"

for file in os.listdir(filepath):
    if file.startswith("Be"):
        print(file)
    else:
        print("The other files: {}".format(file))
finished = True
print("\n")

list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(".txt.mpa")]
print("These are the raw files", list_of_raw_files)

with open(filepath + "\\" + list_of_raw_files[1], 'r') as file:
    lines = file.readlines()
    if lines ==[]:
        print("The file is empty")
    else:
        print("The file is not empty")

header_index = None
for idx, line in enumerate(lines):
    if subject in line:
        header_index = idx
        print(f"Printing the header index of {subject}:", header_index)
        break
if header_index is not None:
    print("The line corresponding to header_index is:", lines[header_index])
else:
    print("No matching header found.")
# Now we have the index of the line we want to start reading from, this is also where the dE/E_final data starts.
# Now we want to read the data from the file and put it into a dataframe.

print(type(lines))

data_index = None
new_lines = lines[header_index:]
for idx, line in enumerate(new_lines):
    if line.startswith("[DATA]"):
        data_index = idx
        print(f"Printing the data index of [DATA]:", data_index)
        break

data_start = data_index + 1

#good way forthis data sheet as after every data there is text so it stops at the first text
data = []
for line in new_lines[data_start:]:
    if any(char.isalpha() for char in line):
        break
    data.append(line.strip().split())

print("checking last element in list",data[-1])
print("checking first element in list",data[0])

# Now we want to put the data into a dataframe
data = pd.DataFrame(data, columns=["E_final", "dE", "counts"])
data["dE"] = pd.to_numeric(data["dE"])
data["E_final"] = pd.to_numeric(data["E_final"])
data["counts"] = pd.to_numeric(data["counts"])
print(data.head())

plt.scatter(data["E_final"], data["dE"], c=data["counts"], cmap='viridis', s=1)  # Adjust the 's' parameter for dot size
plt.xlabel("E_final [keV]")
plt.ylabel("dE [keV]")
plt.colorbar()
plt.show()
print(data["counts"])

plt.scatter(data["dE"], data["counts"], s=1)
plt.scatter(data["E_final"], data["counts"], s=1)
plt.show()


