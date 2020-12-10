import glob
import os
def remove_pkl():
    files = glob.glob('*.pkl')
    for file in files:
        os.remove(file)
remove_pkl()