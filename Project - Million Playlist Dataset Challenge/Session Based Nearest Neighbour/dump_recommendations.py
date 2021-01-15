"""
    This file is the final stage of the pipeline.
    It recommends the tracks per qpid such that
    1) All fields are comma separated
    2) first line: "team_info", <team_name>, <contact_email_addr>
    3) next 500 lines: qpid, track_uri1, ... , track_uri500
        - tracks provided as seed tracks in challenge set must not be included
        - no duplicate tracks in recommended tracks in a qpid
        - exactly 500 tracks per qpid

    command line arguments:
    1) challenge_set_filtered.npy   :   {qpid : numpy.ndarray(track_uris)}
    2) top500_tracks.npy            :   numpy.ndarray(track_uris)
    3) track_scores.json            :   {qpid : {track_uri : score}}

    output file:
    1) recommendations.csv
"""

import json
import numpy as np
import sys
import csv

def error():
    if len(sys.argv) != 4:
        print("ERROR: commandline argument expected")
        exit(0)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def loadFromNpy(filename):
    """ returns the dictionary numpy array data 
        type is chosen for memory efficiency.
    """
    return np.load(filename, allow_pickle = True)[()]

def dumpRecommendations(challenge_set, top500_tracks, track_scores):
    """
        for qpid in track_scores, this function recommends top500
        tracks with highest corresponding score. If there are less
        than 500 tracks then append most popular tracks such that
        1) there are no duplicates
        2) songs does not appear in seed track of qpid in challenge set
        first line of the file should be:
        team_info, <team_name>, <contack_email_address>
    """
    with open("recommendations.csv", 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # example use: writer.writerow([1, "hello", "csv"])
        requires_additional_tracks = 0
        for qpid in track_scores:
            row = [k for k, v in sorted(track_scores[qpid].items(), key = lambda item: (-1)*item[1])[:500]]
            if len(row) == 500:
                writer.writerow([qpid] + row)
            else:
                remaining = 500 - len(row)
                seed_tracks = challenge_set[int(qpid)]
                additional_tracks = []
                for top_track in top500_tracks:
                    if (remaining == 0): break
                    if not top_track in seed_tracks:
                        if not top_track in row:
                            additional_tracks = additional_tracks + [top_track]
                            remaining -= 1
                if (remaining != 0):
                    print("ERROR: can not find enough tracks to append")
                    exit(0)
                writer.writerow([qpid] + row + additional_tracks)
                requires_additional_tracks += 1
            print(qpid, "qpid recommended...")
        print(requires_additional_tracks, "qpids require additional tracks to recommend 500 tracks")

if __name__ == "__main__":
    error()
    challenge_set_filtered_path = sys.argv[1]
    top500_tracks_path = sys.argv[2]
    track_scores_path = sys.argv[3]

    print("fetching datastructures from disk...")
    challenge_set = loadFromNpy(challenge_set_filtered_path)
    top500_tracks = loadFromNpy(top500_tracks_path)
    print('fetching track_scores from disk...')
    track_scores = fetchDict(track_scores_path)
    
    dumpRecommendations(challenge_set, top500_tracks, track_scores)
