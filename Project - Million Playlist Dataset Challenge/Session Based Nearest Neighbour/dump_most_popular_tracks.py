"""
    This file dumps most popular 500 songs on disk
    popularity is calculated by number of playlists
    a track appears in.

    command line arg:
    1) invidx.json : {track_uri : {pid : tf}}

    output file
    1) top500_tracks.npy
"""

import sys
import json
import numpy as np

def error():
    if (len(sys.argv) != 2):
        print("ERROR: commandline argument expected")
        exit(0)

def dumpToNpy(dt, filename):
    """ dump dictionary dt to npy file"""
    np.save(filename, dt)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

if __name__ == '__main__':
    error()
    invidx_path = sys.argv[1]
    print("fetching invidx...")
    invidx = fetchDict(invidx_path)

    print("Calculating track popularity...")
    track_popularity = {k : len(v) for k, v in invidx.items()}
    del invidx

    print("fetching top500 most popular songs")
    top500_tracks = np.array(\
            [k for k, v in sorted(track_popularity.items(), key = lambda item: (-1)*item[1])[:500]])

    dumpToNpy(top500_tracks, "top500_tracks.npy")
