"""
This file takes path to the challenge_set.json.
From the challenge set, we are only interested in
{ pid : Array(track_uri) }
"""

import sys
import json
import numpy as np

def error(lst):
    if (len(lst) != 2):
        print("ERROR: command line args exptected")
        exit(1)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def dumpToNpy(dt, filename):
    """ dump dictionary dt to npy file"""
    np.save(filename, dt)

def loadToNpy(filename):
    return np.load(filename, allow_pickle = True)[()]

def dumpPlaylists(playlists):
    ps = {}
    for playlist in playlists:
        no_tracks = playlist['num_samples']
        if (no_tracks == 0): continue
        pid = playlist['pid']
        tracks = playlist['tracks']
        # space efficient numpy arrays for track_uri
        track_arr = np.empty(no_tracks, dtype = 'object')
        i = 0
        for track in tracks:
            track_arr[i] = track['track_uri']
            i += 1

        ps[pid] = track_arr
    print(len(ps))
    dumpToNpy(ps, "challenge_set_filtered.npy")

if __name__ == "__main__":
    error(sys.argv)
    challenge_set_path = sys.argv[1]
    
    #challenge set
    challenge_set = fetchDict(challenge_set_path)
    playlists = challenge_set['playlists']
    del challenge_set # only playlists are imp
    dumpPlaylists(playlists)

