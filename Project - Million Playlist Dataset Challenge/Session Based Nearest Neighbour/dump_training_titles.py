"""
    This file dumps: numpy.ndarray(title_tokens)
    on disk. This data-structure will be used in
    string matching algorithm to find the nearest
    playlists from the training data.

    command line arguments:
    1) path to training slices

    output file:
    1) training_titles.npy : numpy.ndarray( numpy.ndarray(word))
"""

import sys
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

def error():
    if len(sys.argv) != 2:
        print("ERROR: commanline arguments expected")
        exit(0)

def dumpToNpy(dt, filename):
    """ dump dictionary dt to npy file"""
    np.save(filename, dt)

def fetchDict(filepath):
    with open(filepath, 'r') as file_obj:
        dict_obj = json.load(file_obj)
    return dict_obj

TOTAL_DOCS = 1000000
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


def dumpTrainingTitles(slices_path):
    """ this file dumps: numpy.ndarray(title_strings)"""
    slices = [join(slices_path, f) for f in listdir(slices_path) if isfile(join(slices_path, f))]
    training_titles = np.empty(TOTAL_DOCS, dtype = object)
    file_count = 1
    for f in slices:
        print(f, file_count)
        file_count += 1
        with open(f, 'r') as file_obj:
            data = json.load(file_obj)
            playlists = data['playlists']
            for playlist in playlists:
                pid = playlist['pid']
                title = playlist['name']
                tokens = preprocess(title)
                training_titles[int(pid)] = tokens
    print(training_titles[:10])
    dumpToNpy(training_titles, "training_titles.npy")

if __name__ == '__main__':
    error()
    slices_path = sys.argv[1]
    dumpTrainingTitles(slices_path)
