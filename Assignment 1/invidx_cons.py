import itertools
import os
import re
import time
import sys
import json
import xml.etree.ElementTree as ET
from string import punctuation

import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet

cachedStopWords = stopwords.words("english")


def check_valid_token(token):
    # @todo: Make it better
    return token not in cachedStopWords and token not in ['lrb', 'lcb', 'lsb', 'rrb', 'rcb', 'rsb'] and not re.search(
        '[0-9]+', token)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_and_parse():
    N_docs = 0
    index = {}
    inv_index = {}
    doc_start = time.time()
    # lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
    for file in train_files:
        file_path = os.path.join(sys.argv[1], file)
        if os.path.isfile(file_path):
            with open(file_path) as f:
                end = time.time()
                print({f}, {end - doc_start})
                doc_start = time.time()
                text = f.read()
                text = re.sub("['&\"\'\!]", ' ', text)
                text = '<r>' + text + '</r>'
                root = ET.fromstringlist(text)
                docs = root.getchildren()
                for doc in docs:
                    doc_id = doc.find("DOCNO").text.strip()
                    doc_text = doc.find("TEXT")
                    index[doc_id] = {}
                    # This line handles cases like NEW YORK being tagged seperately as locations. @TODO: make it better
                    # data = re.sub('[\-\+\\\?\;\:\,\+\*\_]+', ' ', data).split()
                    data = str(ET.tostring(doc.find('TEXT')))
                    data = data.translate(str.maketrans('', '', '[]{}(),;:+*\"?$`.'))
                    data = data.translate(str.maketrans('-', ' '))
                    data = re.sub(
                        '(</PERSON>\s*<PERSON>)|(</LOCATION>\s*<LOCATION>)|(</ORGANI[SZ]ATION>\s*<ORGANI[SZ]ATION>)',
                        '', data)

                    # print(data)
                    data = data.translate(str.maketrans('', '', '[]{}(),;:+*_\\\"?$`.'))
                    data = data.split()
                    # if len(data) == 2:
                    #     continue
                    N_docs += 1
                    data = data[1:(len(data) - 1)]
                    last_tag = "</>"
                    token_flag = False
                    # tagged = nltk.pos_tag(data)
                    # for token, tag in tagged:
                    for token in data:
                        if len(token) <= 1:
                            continue
                        if token[0] == '<':
                            last_tag = token
                            if token[1] != '/':
                                token_flag = True
                            else:
                                token_flag = False
                            continue
                        token = token.lower().strip()
                        if not check_valid_token(token):
                            continue
                        # print(token)
                        if token_flag:
                            # token = lemmatizer.lemmatize(token, get_wordnet_pos(tag))
                            # token = stemmer.stem(token)
                            # named_entity += token + " "
                            named_token = token + "_" + last_tag[1]
                            index[doc_id][named_token] = index[doc_id].get(named_token, 0) + 1
                            # if named_token not in inv_index:
                            #     inv_index[named_token] = [doc_id]
                            # else:
                            #     inv_index[named_token] = inv_index.get(named_token).append(doc_id)
                            if named_token not in inv_index:
                                inv_index[named_token] = {doc_id: 1}
                            else:
                                inv_index[named_token][doc_id] = inv_index[named_token].get(doc_id, 0) + 1
                        else:
                            # token = lemmatizer.lemmatize(token, get_wordnet_pos(tag))
                            # token = stemmer.stem(token)
                            index[doc_id][token] = index[doc_id].get(token, 0) + 1
                            # if token not in inv_index:
                            #     inv_index[token] = [doc_id]
                            # else:
                            #     inv_index[token] = inv_index.get(token).append(doc_id)
                            if token not in inv_index:
                                inv_index[token] = {doc_id: 1}
                            else:
                                inv_index[token][doc_id] = inv_index[token].get(doc_id, 0) + 1
                inv_index["$doc_size$"] = {"$doc_size$": N_docs}

    return index, inv_index


def generate_posting_lists(index, inv_index):
    with open(sys.argv[2] + ".dict", "w") as f:
        json.dump(index, f, sort_keys=True)
    with open(sys.argv[2] + ".idx", "w") as f:
        json.dump(inv_index, f, sort_keys=True)


# D:\Downloads\Repos\IRA1\TaggedTrainingAP indexfile
# D:\Downloads\Repos\IRA1\test_run test_indexfile
if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('tokenizers/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    train_files = [f for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]
    print(train_files)

    start = time.time()

    idx, inv_idx = preprocess_and_parse()

    generate_posting_lists(idx, inv_idx)

    stop = time.time()

    print(stop - start)
