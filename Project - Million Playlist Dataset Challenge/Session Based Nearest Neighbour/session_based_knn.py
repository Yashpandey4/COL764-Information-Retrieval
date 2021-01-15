"""
This file implements session-based nearest neighbor
algorithm.
command line args: 
    <training_directory_path>   : this directory must contain mpd.slice.*.json files
    invidx.json                 : {track_uri: {pid: freq}}          // must be in pwd
    modifications.json          : numpy.ndarray(last_modified)      // must be in pwd
    challenge_set_filtered.npy  : {pid: numpy.ndarray(track_uri)}   // must be in pwd
"""

import sys
import json
import numpy as np

def error(args):
    if (len(args) != 5):
        print("ERROR: commandline args expected.")
        exit(0)

def sessionBasedKNN(training_path, invidx, \
                      modifications, challenge_set_path):
    print("hello world")

def fetchDict(filename):
    with open(filename) as file_obj:
        data = json.load(file_obj)
    return data

def loadFromNpy(filename):
    """ returns the dictionary: {pid : numpy.ndarray(track_uri)}
        numpy array data type is chosen for memory efficiency.
    """
    return np.load(filename, allow_pickle = True)[()]

if __name__ == "__main__":
    print()
    # argument handling
    error(sys.argv)
    training_path = sys.argv[1]
    invidx_filename = sys.argv[2]
    modification_filename = sys.argv[3]
    challenge_set_filename = sys.argv[4]

    # filtered challenge set
    print("Loading filtered challenge_set to memory...")
    challenge_set = loadFromNpy(challenge_set_filename)
    print(len(challenge_set))

    # inverted index
    print("Loading inverted index to memory...")
    invidx = fetchDict(invidx_filename)
    print(len(invidx))

    # modifications
    print("Loading modifications to memory...")
    modifications = loadFromNpy(modification_filename)
    print(len(modifications))
