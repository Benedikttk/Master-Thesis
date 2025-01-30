from EXYZReader import read_exyz_file
import numpy as np
import matplotlib.pyplot as plt

# I think what i want to do is see what the optimal gas density for Isobutan is:
# The way this will be checked is by Counting how many Ions cross from the first anode(deltaE) and reach the second anode(E_res)
# Something important is that if they travel more than 315 mm they will surpase the second anode, which is something we dont want.
# So maybe fitting soemthing tom the points, with soem condition???? mhmmmm

#trajectories of ions
file_path = r"C:\Users\benja\Desktop\Speciale\Data\EXYZs\EXYZ1.txt"  
df = read_exyz_file(file_path)

#print(IonNumbers)

#test af indeces
indices_list = df[df["Ion Number"] == 1].index.tolist()

print(indices_list)