import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'

# Idea: https://www.w3schools.com/python/python_ml_getting_started.asp Evrything is bassed from here.
# - We want to read the data from the files in the filepath.
# - We want to construct a delta E E_res plot for the data.
# - We want to calculate the counts of the Be10 ions within a reagon of interest
# and see if we can distigguse between the B10 ions.
# - We want to use k-neerest neighbour to predict the R_n value. and find the best k value. 
# - We also want it to be ale to outline the region of interest.
# - We want to be able to calculate the R_n value from the data.
# - Finally we want to compare with the R_n value from the AARAMS model.

#Task 1#
# Check if the file is a AARAMS file 
# and read values from the fil(s) in the directory get them into a dataframe.

for file in os.listdir(filepath):
    if file.startswith("Be"):
        print(file)
    else:
        print("The other files: {}".format(file))
finished = True
print("\n")

# Begining task 1 - Reading the data from the files
# I think i just want to start with the first file and then see if i can make a function that can do it for all the files.
# I will start by reading the first file.

# We want the Name = dE and the Name = E final values.
# So we will be looking for these lins and then searching for [Data] and then reading the data from there.

list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(".txt.mpa")]
print("These are the raw files", list_of_raw_files)

# Now we want top open the file and read from it 

names = ["dE", "E final"]
with open(filepath + "\\" + list_of_raw_files[0], 'r') as file:
    lines = file.readlines()
    if lines ==[]:
        print("The file is empty")
    else:
        print("The file is not empty")

header_index = None
header_index_list = []
for idx, line in enumerate(lines):
    for name in names:
        if f"NAME={name}" in line and line.strip() == f"NAME={name}": #filters away stuff like NAME=dE + E final
            header_index = idx
            header_index_list.append(header_index)
            print(f"Printing the header index of [NAME={name}]:", header_index)
            break

# Print the specific line corresponding to header_index
if header_index is not None:
    print("The line corresponding to header_index is:", lines[header_index])
else:
    print("No matching header found.")


# Now we want to find the [Data] line and read the data from there.
# I think what i want to do is to read the data from the first header_index to the next header_index
# this is going to be ugly!!

#sorry 

#thinking thinking, so i have the idx of wha???

#print(lines[header_index_list[0]])
#print(lines[header_index_list[1]])

# Extract the data between the header indices
dummy_list_one_dE = lines[header_index_list[0]:header_index_list[1]]

# Find the [DATA] line and read the data from there for dE
data_start_index_dE = None
for idx, line in enumerate(dummy_list_one_dE):
    if "[DATA]" in line:
        data_start_index_dE = idx + 1
        break

if data_start_index_dE is not None:
    data_lines_dE = dummy_list_one_dE[data_start_index_dE:]
    data_dE = [line.strip().split() for line in data_lines_dE]
    df_dE = pd.DataFrame(data_dE, columns=["dE", "Counts"])
    df_dE["dE"] = pd.to_numeric(df_dE["dE"], errors='coerce')
    df_dE["Counts"] = pd.to_numeric(df_dE["Counts"], errors='coerce')
    df_dE = df_dE.dropna()  # Drop rows with NaN values

    print(df_dE.head())
else:
    print("No [DATA] section found in the file for dE.")


dummy_list_two_E_final = lines[header_index_list[1]:header_index_list[1]+(header_index_list[1] - header_index_list[0])]
data_start_index_E_final = None
for idx, line in enumerate(dummy_list_two_E_final):
    if "[DATA]" in line:
        data_start_index_E_final = idx + 1
        break
print("idx of E Final", data_start_index_E_final)

if data_start_index_E_final is not None:
    data_lines_E_final = dummy_list_two_E_final[data_start_index_E_final:]
    data_E_final = [line.strip().split() for line in data_lines_E_final]
    df_E_final = pd.DataFrame(data_E_final, columns=["E final", "Counts"])
    df_E_final["E final"] = pd.to_numeric(df_E_final["E final"], errors='coerce')
    df_E_final["Counts"] = pd.to_numeric(df_E_final["Counts"], errors='coerce')
    df_E_final = df_E_final.dropna()  # Drop rows with NaN values

    print(df_E_final.head())
else:
    print("No [DATA] section found in the file for E final.")

# Now we have the data from the first file.
# We want to plot the data.
# We want to make a function that can do this for all the files.
# But first we want to check if we can plot these two dataframes.

plt.plot(df_dE["dE"], df_dE["Counts"], label="dE")  
plt.plot(df_E_final["E final"], df_E_final["Counts"], label="E final")
plt.show()

print("leanght of dE", len(df_dE))
print("leanght of E final", len(df_E_final))

