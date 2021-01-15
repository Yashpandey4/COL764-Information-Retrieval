"""
    This file dumps playlist -> tracks for playlists that
    appears in topm playlists per query

    commandline arg:
    1) topm.npy : {qpid : numpy.ndarray(pids)}
    2) <path_to_original_training_data_dir>
    3) invidx.json : {track_uri : { pid : freq}}
    4) tracks_per_pid.npy : numpy.ndarray(#tracks)

    dump file:
    topm_playlists_to_tracks.npy : {pid : {track_uri : tfidf}}

    Hyper parameters
    1) S := used for smoothing in term freq
"""

import sys
import numpy as np
import json
import os.path
from os import listdir
from os.path import isfile, join
import math

TOTAL_DOCS = 1000000
""" Hyper parameter """
S = 10 # used for smoothing in term freq

def error(args):
    if (len(args) != 5):
        print("ERROR: Commandline arguents expected")
        exit(1)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def loadFromNpy(filename):
    """ returns the dictionary: {pid : numpy.ndarray(track_uri)}
        numpy array data type is chosen for memory efficiency.
    """
    return np.load(filename, allow_pickle = True)[()]

def getSet(topm):
    """ topm:   {qpid : numpy.ndarray(pid)}
        return union of topm playlists"""
    topm_pids = set()
    for qpid in topm:
        topm_pids.update(topm[qpid])
    return topm_pids

def dumpToNpy(dt, filename):
    dir_name = "pid_vectors"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    np.save(filename, dt)

    os.chdir("../")

def dumpToJson(dt, filename):
    """ dump dictionary dt to json file"""
    dir_name = "pid_vectors"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    with open(filename, 'w') as f_obj:
        json.dump(dt, f_obj)
    dir_name = "pid_vectors"

    os.chdir("../")

def dumpPidVector(slices_path, topm_pids, invidx, tracks_per_pid):
    """
        store following
        pid -> numpy.ndarray(track_uri)
        for all pid in topm_pids
    """
    prefix = "/home/dwijesh/Documents/sem7/ir/dataset/project/spotify_million_playlist_dataset/data/mpd.slice."
    suffix = ".json"
    pids_vectorized = 0
    for counter in range(1000):
        f = prefix + str(counter*1000) + "-" + str(counter*1000 + 999) + suffix
        pid_vectors = {}
        outfile_name = str(counter*1000) + "_pid_vectors.json"
        with open(f, 'r') as file_obj:
            data = json.load(file_obj)
            playlists = data['playlists']
            del data

            for playlist in playlists:
                pid = playlist['pid']
                if not pid in topm_pids: continue
                pid_vectors[pid] = {}
                track_uris = np.array([track['track_uri'] for track in playlist['tracks']])
                for track_uri in track_uris:
                    idf = math.log(TOTAL_DOCS/len(invidx[track_uri]))
                    tf = invidx[track_uri][str(pid)]/(tracks_per_pid[pid] + S)
                    pid_vectors[pid][track_uri] = tf * idf
        pids_vectorized += len(pid_vectors)
        dumpToJson(pid_vectors, outfile_name)
        print("slice: ", counter, " done...")
    print(pids_vectorized, " pids vectorized...")

if __name__ == "__main__":
    error(sys.argv)
    topm_filename = sys.argv[1]
    slices_path = sys.argv[2]
    invidx_filename = sys.argv[3]
    tracks_per_pid_path = sys.argv[4]
    
    invidx = fetchDict(invidx_filename)
    topm = loadFromNpy(topm_filename)
    tracks_per_pid = loadFromNpy(tracks_per_pid_path)
    print("read datastructures to the memory...")
    topm_pids = getSet(topm) # int pids
    del topm
    print(len(topm_pids), " pids needs to be vectorized...")

    dumpPidVector(slices_path, topm_pids, invidx, tracks_per_pid)
