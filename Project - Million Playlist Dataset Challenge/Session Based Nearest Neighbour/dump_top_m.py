"""
    This file dumps (set of) top-m playlists per
    pid of challenge set
    commandline arguments are:
    - invidx.json                   : {track_uri : {pid : freq}}
    - modifications.npy             : numpy.ndarray(lastmodified)
    - challenge_set_filtered.npy    : {pid : numpy.ndarray(track_uri)}

    for each challenge playlist, find set of m
    neighboring playlists and dump it on disk

    HyperParameter:
    1) M := How many neighbors to choose per query pid
"""

import sys
import json
import numpy as np

# hyper parameter
M = 4000

def error(args):
    if (len(args) != 4):
        print("ERROR: Command line args expected")
        exit(1)

def fetchDict(filename):
    with open(filename) as file_obj:
        data = json.load(file_obj)
    return data

def loadFromNpy(filename):
    """numpy array data type is chosen for memory efficiency."""
    return np.load(filename, allow_pickle = True)[()]

def dumpTopM(invidx, modifications, challenge_set):
    topm = {}
    # playlists of challenge set
    qpid_count = 1
    for qpid in challenge_set:
        print(qpid_count, 'the qpid processing...')
        qpid_count += 1
        qtracks = challenge_set[qpid]
        # set of playlist having atleast one 
        #track in common with query playlist
        neighbors = set()
        for qtrack_uri in qtracks:
            neighbors.update(set(list(invidx[qtrack_uri].keys())))
        neighbors = [int(n) for n in neighbors]
        
        # subsampling playlists with latest edit made since epoch
        time_pid = [(-modifications[i], neighbors[i]) for i in range(len(neighbors))]
        del neighbors
        # first element of tuple is kept negative to sort
        # tuples in descending order of thier original magnitude
        time_pid = sorted(time_pid)
        # now choosing top m
        time_pid = time_pid[:M]

        # storing: qpid -> numpy.ndarray(pids)
        topm[qpid] = np.array([pid for (_, pid) in time_pid])
    # dumping to disk
    np.save("topm.npy", topm)

if __name__ == "__main__":
    error(sys.argv)
    invidx_filename = sys.argv[1]
    modifications_filename = sys.argv[2]
    challenge_set_filename = sys.argv[3]

    print("Loading invidx to memory...")
    invidx = fetchDict(invidx_filename)
    print("Loading modifications to memory...")
    modifications = loadFromNpy(modifications_filename)
    print("Loading challenge set to memory")
    challenge_set = loadFromNpy(challenge_set_filename)

    dumpTopM(invidx, modifications, challenge_set)
