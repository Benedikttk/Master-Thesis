import os
import pandas as pd
import numpy as np
filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'

#Task 1#
# Check if the file is a AARAMS file -> Done
# print file names in the directory -> Done
#Task 2#
# The normal .txt files have already calculated the number of Be10, we can do fst calculations with them
# These files are the once we should satrt with to calculate R_n
# Extract values from the files and put them in a dataframe -> Done
# Calculate the average Be10 ion count and the uncertanty, as it is just counting statistics -> Done
# Calculate the current of Be10 -> Done
# Calculate the find the number of Be9, Be9 is given as a current, so we can calculate the number of Be9 ions

#Task 3#
# The other files, .txt.mpa, are also txt files just with moreinformation/
# These files are the once we should use to calculate the R_n with the AARAMS model and see if we can impliment/
# k nearest neighbour to predict the R_n value. 

for file in os.listdir(filepath):
    if file.startswith("Be"):
        print(file)

# Begining task 1
# I am only interested in txt files to start with as they have already calculatd the Be10 ions

list_of_files = [file for file in os.listdir(filepath) if file.endswith(".txt")]
list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(".txt.mpa")]

#print("These are the files with calculated values of Be10", list_of_files)
#print("These are the raw files", list_of_raw_files)

#Now we want to open the files and read the data from them
with open(filepath + "\\" + list_of_files[0], 'r') as file:
    lines = file.readlines()
    if lines ==[]:
        print("The file is empty")
    else:
        print("The file is not empty")
for idx, line in enumerate(lines):
    if line.strip().startswith("[BLOCK DATA]"):
        header_index = idx
        print("printing the header index: of [BLOCK DATA]:",header_index)
        break

data_lines = lines[header_index + 1:]
#print("The data lines are:", data_lines)

column_names = [
    "Blk", "10Becnts", "Totalcnts", "10Bcnts", "cnts", "cnts", \
        "LiveTime", "10Becps", "9Becur", "nonecur", "9BeOcur", "nonecur",\
            "10Be/9Be", "10Be/none", "9Be/none", "9Be/9BeO", "none/none",\
                "9BeO/none", "TargetCur", "Flags"
]

# Convert to DataFrame
df = pd.DataFrame([line.split() for line in data_lines], columns=column_names)

#Convort the relative coloumns to numerikal values
df = df.apply(pd.to_numeric, errors='ignore')
df = df.drop(index=0)
#Now we want to calculate the average Be10 ion count and the uncertanty, as it is just counting statistics
#The average is the mean value of the Be10 counts and the uncertanty is the standard deviation

# Print the dimensions of the "10Becnts" column

IsItANumber = pd.to_numeric(df["10Becnts"], errors='coerce')

if IsItANumber.isnull().values.any():
    print("The column contains non-numeric values")
else:
    print("The column contains only numeric values")
#what am I doing here? why is this so hard!!!!!!!!!!!!!!!!!!
Be10cnts = df["10Becnts"].astype(float)

avg_Be10cnts = Be10cnts.mean()
std_Be10cnts = Be10cnts.std()

print(f"The average Be10 counts is {avg_Be10cnts} and the standard deviation is ± {std_Be10cnts}")

#Now i want it in Amperes, we have the life time of the measurment, so we can calculate the current by\
#I = Q/t, where Q is the charge and t is the life time of the measurment

average_time = pd.to_numeric(df["LiveTime"], errors='coerce').astype(float).mean()
average_time_uncertanty = pd.to_numeric(df["LiveTime"], errors='coerce').astype(float).std()
Q_Be10 = 1.6e-19 * avg_Be10cnts
I_Be10 = Q_Be10 / average_time * 10**6 #The current is in microamperes
I_Be10_uncertanty = Q_Be10 / average_time**2 * average_time_uncertanty
print(f"The current of Be10 is {I_Be10} and the uncertanty is ± {I_Be10_uncertanty} [micro A]") #not sure if this is correct

# Now we want to find the number of Be9, Be9 is given as a current, so we can calculate the number of Be9 ions
# For this we need to find the detectror live time, the current of Be9 

name_of_info = "Detector live time [s]"
with open(filepath + "\\" + list_of_files[0], 'r') as file:
    lines = file.readlines()
    if lines ==[]:
        print("The file is empty")
    else:
        print("The file is not empty")
    for idx, line in enumerate(lines):
        if line.strip().startswith(name_of_info):
            detector_live_time = line.strip().strip(":").split(":")[1]
            break
        
detector_live_time = pd.to_numeric(detector_live_time, errors='coerce').astype(float)

name_of_info = "9Be current [A]"
with open(filepath + "\\" + list_of_files[0], 'r') as file:
    lines = file.readlines()
    if lines ==[]:
        print("The file is empty")
    else:
        print("The file is not empty")
    for idx, line in enumerate(lines):
        if line.strip().startswith(name_of_info):
            Be9_current = line.strip().strip(":").split(":")[1]
            break
print("The detector live time is:", detector_live_time, "[s]")
print("The current of Be9 is:", Be9_current,"[A]")

Be9_current = pd.to_numeric(Be9_current, errors='coerce').astype(float) #remember its only charge staet q=1

#Using I=Q/t=q*n/t -> n = I*t/q

Be9cnts = Be9_current * detector_live_time / 1.6e-19
print(f"The number of Be9 ions is {Be9cnts}")

#Now we can calculate the ratio of Be10/Be9
# I think because we have 10 blocks that I have to multiply with 10 so the ratio is correct
#The uncertainty is given by the formula sqrt((std_Be10cnts/avg_Be10cnts)^2 + (Be9cnts/Be9_current)^2)
#Now I have to find the total number of Be10, I will just take this as the mean * 10 and now I can calculate the correct uncertainty also
sumed_avg_Be10cnts = avg_Be10cnts * 10
R_n = sumed_avg_Be10cnts / Be9cnts
R_n_uncertainty = R_n * np.sqrt((std_Be10cnts / sumed_avg_Be10cnts)**2 + (0 / Be9cnts)**2)
print(f"The ratio of Be10/Be9 is {R_n} and the uncertainty is ± {R_n_uncertainty}")
R_nominiel = 27.1e-12
R_nominiel_uncertainty = 0.3e-12
isotropic_ratio_efficiency = R_n / R_nominiel * 100
isotropic_ratio_efficiency_uncertainty = isotropic_ratio_efficiency * np.sqrt((R_n_uncertainty / R_n)**2 + (R_nominiel_uncertainty / R_nominiel)**2)
print(f"The isotropic ratio efficiency is {round(isotropic_ratio_efficiency,3)} ± {round(isotropic_ratio_efficiency_uncertainty,3)} %")
