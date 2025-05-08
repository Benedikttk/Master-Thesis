import os

from Functions import FileCheck

Filepath = r'C:\Users\benja\Desktop\Data_nye\2025_04_25_Blk@850'

files = [i for i in FileCheck(filepath=Filepath, endswith=".mpa")]

print(files)