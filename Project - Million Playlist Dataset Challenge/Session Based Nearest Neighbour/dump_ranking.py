"""
    This file dumps <=top500 tracks for each qpid.

    command line arguments:
    1) pairwise_similarity.json : {qpid : {pid : sim(qpid, pid)}}
    2) challenge_set_filtered.npy : {qpid : numpy.ndarray(track_uri)}
    3) pid_vectors : directory path to {pid : {track_uri : tfidf}}

    outfile:
    1) recommendations.npy: {qpid : numpy.ndarray(track_uri)}
"""

import sys
import numpy as np
import json
import time
from multiprocessing import Pool

def error(args):
    if (len(args) != 4):
        print("ERROR: command line arguments expected")
        exit(1)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def loadFromNpy(filename):
    """ returns the dictionary numpy array data 
        type is chosen for memory efficiency.
    """
    return np.load(filename, allow_pickle = True)[()]

def dumpToJson(dt, filename):
    """ dump dictionary dt to json file"""
    with open(filename, 'w') as f_obj:
        json.dump(dt, f_obj)

def dumpRanking(similarity, challenge_set, pid_vectors_path):
    scores = {qpid : {} for qpid in similarity}
    # bringing all 1000 slices to memory is not possible
    iterations = 100 # bring slices in this many iterations
    slices_per_iteration = 1000//iterations
    for i in range(iterations):
        print("iteration:", i, "running...")
        start = i*slices_per_iteration
        end = (i+1)*slices_per_iteration # not inclusive
        slices = np.empty(end-start, 'object')
        for slice_id in range(start, end, 1):
            slice_filename = pid_vectors_path + "/"+ str(slice_id*1000) + "_pid_vectors.json"
            slices[slice_id-start] = fetchDict(slice_filename)
            print("slice", slice_id, "fetched...")

        # slices are ready: {pid : {track_uri : tfidf}}

        qpid_counter = 0
        for qpid in similarity: #type(qpid) = str
            """collect set of tracks from all neighboring playlists 
            of qpid. excluding tracks that already appears in qpid."""
            # collecting valid tracks
            eligible_track_uris = set()
            neighbor_pids = list(similarity[qpid].keys())
            # not all playlists from neighbor_pids lie
            # between start and end
            eligible_pids = [p for p in neighbor_pids if ((int(p)//1000 >= start) and (int(p)//1000 < end))]

            for neighbor_pid in eligible_pids: # type(neighbor_pid) = str
                neighbor_pid_index = int(neighbor_pid)//1000
                eligible_track_uris.update(list(slices[neighbor_pid_index-start][neighbor_pid].keys()))
            qtrack_uris = set(challenge_set[int(qpid)])
            eligible_track_uris -= qtrack_uris

            for eligible_track_uri in eligible_track_uris:
                if not eligible_track_uri in scores[qpid]:
                    scores[qpid][eligible_track_uri] = 0.0 

            # eligible_track_uris are ready: set(track_uri)
            # this set of tracks exclude tracks that already
            # exists in query playlist
            for eligible_track_uri in eligible_track_uris:
                score = sum(similarity[qpid][target_pid] \
                                for target_pid in eligible_pids \
                                    if (eligible_track_uri in slices[(int(target_pid)//1000)-start][target_pid])\
                            )
                scores[qpid][eligible_track_uri] += score
            qpid_counter += 1
            if (qpid_counter % 1000 == 0): print(qpid_counter, "qpids done...")
    dumpToJson(scores, "track_scores.json")

if __name__ == "__main__":
    error(sys.argv)
    similarity = fetchDict(sys.argv[1])
    challenge_set = loadFromNpy(sys.argv[2])
    print("data-structures fetched to memory...")
    pid_vectors_path = sys.argv[3]

    dumpRanking(similarity, challenge_set, pid_vectors_path)
