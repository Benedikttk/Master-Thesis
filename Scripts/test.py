from ReadingFromSRIMfile import process_file
import numpy as np  
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_path = r"C:\Users\benja\Desktop\Speciale\Data\EXYZs\EXYZ1.txt"  

# Check if the file exists
if os.path.isfile(file_path):
    print(f"The file at {file_path} exists.")
else:
    print(f"The file at {file_path} does not exist.")


txt = "hello, my name is Peter, I am 26 years old"

x = txt.split(", ")

print(x)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the background image
img = mpimg.imread(r'C:\Users\benja\Desktop\Speciale\Billeder\DetectorFigure.png')
fig, ax = plt.subplots()

# Show the background image
ax.imshow(img, extent=[0, 10, 0, 10])  # You can set the extent (x, y range) accordingly

# Plot your data on top of the image
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
ax.plot(x, y, color='red', marker='o')

# Save the new figure plt.savefig('output_with_data.png')

