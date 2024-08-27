import pandas as pd
import numpy as np
import glob
import os 

directory = os.path.dirname(os.path.realpath(__file__))

os.chdir(directory)

files = glob.glob("*.csv", root_dir=directory)

print(directory)
print(files)

for file in files:
    data = np.genfromtxt(file, delimiter=",")
    print(file)
    print(data.shape)
    np.savez(str(os.path.splitext(file)[0]), data)



