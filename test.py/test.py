import os

from Functions import FileCheck, deltE_Efinal

# 
#Filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
Filepath = r'C:\Users\benja\Desktop\Data_nye\2025_04_25_Blk@850'
Subject = "[CDAT0"

files = [i for i in FileCheck(filepath=Filepath, endswith=".txt.mpa")]
