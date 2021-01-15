"""
    This file appends/creates recommendations for the case
    when only playlist name is given.
    
    command line arguments:
    1) title_invidx.json        : {title_token : list(pid)}
    2) challenge_titles.npy     : {qpid : title_tokens}
    3) top500_tracks.npy        : numpy.ndarray(top500 tracks)
    4) pid_vectors              : path to {pid : {track : tfidf}}
    5) path to training data    : directory

    append/create recommendations to:
    1) recommendations.csv
"""

import json
import sys
import csv
import numpy as np
from os import listdir
from os.path import isfile, join

def error():
    if len(sys.argv) != 6:
        print("ERROR: command line arguments expected")
        exit(0)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

def loadToNpy(filename):
    return np.load(filename, allow_pickle = True)[()]

def getNeighborPids(title_invidx, challenge_titles):
    """return {qpid : list(pids)}"""
    neighbor_pids = {}
    for qpid, title_tokens in challenge_titles.items():
        similar_pids = set()
        for token in title_tokens:
            if token in title_invidx:
                similar_pids.update(title_invidx[token])
        neighbor_pids[qpid] = list(similar_pids)
    return neighbor_pids

def dumpTOP500Recommendations(top500_tracks, neighbor_pids):
    """
        write to string_matching_recommendations.csv for the 
        case when there is no neighboring pid for a given qpid
    """
    with open('string_matching_recommendations.csv', 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        for qpid, pids in neighbor_pids.items():
            if(len(pids) == 0):
                writer.writerow([qpid] + top500_tracks.tolist())

def getEligibleTracks(pid_vectors_path, neighbor_pids):
    """
        return {qpid : {track_uri : freq}}
    """
    eligible_tracks = {qpid:{} for qpid, pids in neighbor_pids.items() if len(pids) != 0}
    iterations = 50
    slices_per_iterations = 1000//iterations
    for i in range(iterations):
        print("iteration", i, "running...")
        start = i*slices_per_iterations
        end = (i+1)*slices_per_iterations #not inclusing
        slices = np.empty(end-start, 'object')
        for slice_id in range(start, end, 1):
            slice_filename = pid_vectors_path + '/' + str(slice_id*1000) + "_pid_vectors.json"
            slices[slice_id-start] = fetchDict(slice_filename)
            print("slice", slice_id, "fetched...")

        discovered_pid_vectors = {} # {pid: [track_uris]}
        # type(slices) : numpy.ndarray( { pid : numpy.ndarray(track_uris) } )
        for qpid, pids in neighbor_pids.items():
            print(qpid, " qpid processing...")
            if len(pids) == 0: continue
            eligible_pids = [str(pid) for pid in pids if (int(pid)//1000)>=start and (int(pid)//1000)<end]
            for pid in eligible_pids: #type(pid) = str
                if pid in slices[(int(pid)//1000)-start]: # pid vector already exists in pid_vectors/
                    for track_uri, tfidf in slices[(int(pid)//1000)-start][pid].items():
                        if track_uri in eligible_tracks[qpid]:
                            eligible_tracks[qpid][track_uri] += 1
                        else:
                            eligible_tracks[qpid][track_uri] = 1

                elif pid in discovered_pid_vectors: # we have discovered it
                    for track_uri in discovered_pid_vectors[pid]:
                        if track_uri in eligible_tracks[qpid]:
                            eligible_tracks[qpid][track_uri] += 1
                        else:
                            eligible_tracks[qpid][track_uri] = 1

                else: # fetch pid vector from training data
                    start_index = (int(pid)//1000)*1000
                    end_index = start_index + 999
                    file_path = sys.argv[5] + '/mpd.slice.' + str(start_index) + '-' + str(end_index) + '.json'
                    pid_slice = fetchDict(file_path)
                    playlist = pid_slice['playlists'][int(pid) - start_index]
                    playlist_id = playlist['pid']
                    if (int(playlist_id) != int(pid)):
                        print('ERROR: assumption to index in a playlists is failed')
                        exit(0)
                    all_tracks = [track['track_uri'] for track in playlist['tracks']]
                    discovered_pid_vectors[pid] = all_tracks
                    for track_uri in all_tracks:
                        if track_uri in eligible_tracks[qpid]:
                            eligible_tracks[qpid][track_uri] += 1
                        else:
                            eligible_tracks[qpid][track_uri] = 1

    return eligible_tracks

def dumpRecommendations(eligible_tracks, top500_tracks):
    with open('string_matching_recommendations.csv', 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        for qpid, track_freq in eligible_tracks.items():
            print(qpid, 'qpid recommending...')
            top_recommendations = \
                [track_uri for track_uri, freq in sorted(track_freq.items(), \
                                                    key=lambda item:(-1)*item[1])[:500]]
            if len(top_recommendations) == 500:
                writer.writerow([qpid] + top_recommendations)
            else:
                extra_tracks = [top_track for top_track in top500_tracks if not top_track in top_recommendations]
                remaining = 500 - len(top_recommendations)
                if len(extra_tracks) < remaining:
                    print("ERROR: not enough tracks")
                    exit(0)
                writer.writerow([qpid] + top_recommendations + extra_tracks[:remaining])

if __name__ == '__main__':
    error()
    print('fetching title info from disk...')
    title_invidx = fetchDict(sys.argv[1])
    challenge_titles = loadToNpy(sys.argv[2])
    print('fetching neighboring pids...')
    neighbor_pids = getNeighborPids(title_invidx, challenge_titles)
    del title_invidx
    del challenge_titles

    with open('string_matching_recommendations.csv', 'w', newline='') as outfile:
        print("old content in string_matching_recommendations.csv has been cleared")

    top500_tracks = loadToNpy(sys.argv[3])
    pid_vectors_path = sys.argv[4]
    print('recommending qpids having emtpy neighboring pids...')
    dumpTOP500Recommendations(top500_tracks, neighbor_pids)
    eligible_tracks = getEligibleTracks(pid_vectors_path, neighbor_pids)
    dumpRecommendations(eligible_tracks, top500_tracks)
