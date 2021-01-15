"""
    This file generates invidx for title tokens from
    taining data.

    command line argument:
    1) training_titles.npy      : numpy.ndarray(title_tokens)

    output file:
    1) title_invidx.json        : {token : list(pid)}
"""

import sys
import json
import numpy as np

def error():
    if len(sys.argv) != 2:
        print("ERROR: command line arguments expected")
        exit(0)

def loadFromNpy(filename):
    """ returns the dictionary numpy array data 
        type is chosen for memory efficiency.
    """
    return np.load(filename, allow_pickle = True)[()]

TOTAL_DOCS = 1000000

def dumpToJson(dt, filename):
    """ dump dictionary dt to json file"""
    with open(filename, 'w') as f_obj:
        json.dump(dt, f_obj)

def dumpTitleInvidx(training_titles):
    title_invidx = {}
    for i in range(TOTAL_DOCS):
        for token in training_titles[i]:
            if token in title_invidx:
                title_invidx[token].append(i)
            else:
                title_invidx[token] = [i]
    print('dumping title invidx to disk...')
    dumpToJson(title_invidx, "title_invidx.json")

if __name__ == '__main__':
    error()
    training_titles = loadFromNpy(sys.argv[1])
    print('training titles fetched from disk...')
    dumpTitleInvidx(training_titles)
