"""
    This file dumps: {qpid : title_tokens}
    on disk. This datastructure will be used for
    string matching algorithm. 

    command line argument:
    1) path to challenge_set.json

    output file:
    1) challenge_titles.npy: {qpid : numpy.ndarray(word)}
"""

import sys
import numpy as np
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

def error():
    if len(sys.argv) != 2:
        print("ERROR: command line argument expected")
        exit(0)

def dumpToNpy(dt, filename):
    """ dump dictionary dt to npy file"""
    np.save(filename, dt)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

STOP_WORDS = set(stopwords.words('english'))
PS = PorterStemmer()

def preprocess(title):
    # lower case
    title = title.lower()
    # special char replacement
    title = re.sub('\W+',' ', title)
    # stop word removal
#    tokens = [w for w in word_tokenize(title) if not w in STOP_WORDS]
    tokens = word_tokenize(title)
    # remove tokens of length of 1
#    tokens = [w for w in tokens if len(w) != 1]
    # stemming
    tokens = [PS.stem(w) for w in tokens]

    return np.array(tokens)

def dumpChallengeTitles(playlists):
    challenge_titles = {}
    for playlist in playlists:
        qpid = playlist['pid']
        num_samples = playlist['num_samples']
        if int(num_samples) != 0: continue
        title = playlist['name']
        tokens = preprocess(title)
        challenge_titles[qpid] = tokens
        print(qpid, "qpid title processed...")
    dumpToNpy(challenge_titles, "challenge_titles.npy")
    print(len(challenge_titles), "queries had zero tracks in it")

if __name__ == '__main__':
    error()
    challenge_set_path = sys.argv[1]
    challenge_set = fetchDict(challenge_set_path)
    playlists = challenge_set['playlists']
    del challenge_set
    dumpChallengeTitles(playlists)
