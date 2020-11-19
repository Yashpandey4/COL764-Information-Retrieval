import argparse
import os
import re
import time
import json
from string import punctuation
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
import multiprocessing as mp


# @TODO: TAs, Please remove manually in my submission if I forget to remove it
def set_java_env():
    java_path = "C:/Program Files/Java/jdk-11.0.7/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    nltk.internals.config_java(java_path)


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def convert_lower_case(data):
    return np.char.lower(data)


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def remove_stop_words(data):
    stopwords_dict = Counter(stop_words)
    text = ' '.join([word for word in text.split() if word not in stopwords_dict])


def tag_entities():
    jar = './stanford-ner.jar'
    model = './english.all.3class.distsim.crf.ser.gz'

    ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

    tags = ['PERSON', 'LOCATION', 'ORGANIZATION']
    punc = ['.', ',', ':', ';', '"', '!', '?', '%']
    # write if filename.startswith('ap'):

    for idx, data in enumerate(query_data):
        words_ner = []
        res = ""
        tokenized = nltk.sent_tokenize(data)
        # print(tokenized)
        for i in tokenized:
            wordsList = nltk.word_tokenize(i)
            words_ner = ner_tagger.tag(wordsList)
            for j in words_ner:
                if j[0] not in punc and not (j[0].startswith('\'')):
                    res += " "
                if j[1] in tags:
                    # handle.write("%s\n" % words_ner)
                    res += "<" + j[1] + "> " + j[0] + " </" + j[1] + ">"
                else:
                    res += j[0]
        res = res.strip()
        query_data[idx] = res
        print(res)
    # print(query_data)


def tf_idf_helper(token):
    global doc_query_score, stemmer
    if token not in inv_index:
        similar_words(stemmer.stem(token))
        return
    doc_list_with_token = inv_index[token]
    for doc in doc_list_with_token:
        if doc not in tf_idf:
            tf_idf[doc] = {}
        if token in tf_idf[doc]:
            continue
        else:
            tf = 1 + np.log(inv_index[token][doc])
            idf = np.log(1 + (N_docs / len(inv_index[token])))
            tf_idf[doc][token] = tf * idf
        doc_query_score[doc] = doc_query_score.get(doc, 0) + tf_idf[doc][token]


def similar_words(token, tag=""):
    matches = []
    if not tag:
        matches = list(filter(lambda s: s.startswith(token), list(inv_index.keys())))
    elif tag == 'N':
        matches = list(filter(lambda s: s.startswith(token) and s[-1] in ['P', 'O', 'L'], list(inv_index.keys())))
    else:
        matches = list(filter(lambda s: s.startswith(token) and s[-1] == tag, list(inv_index.keys())))
    for token in matches:
        tf_idf_helper(token)


def calculate_tf_idf(token, named_entity_search, prefix_search, named_entity):
    if named_entity_search:
        if not prefix_search:
            if named_entity != 'N':
                tf_idf_helper(token)
            else:
                for tkn in [token + "_P", token + "_O", token + "_L"]:
                    tf_idf_helper(tkn)
        else:
            if named_entity != 'N':
                similar_words(token, named_entity)
            else:
                for tag in ['P', 'O', 'L']:
                    similar_words(token, tag)
    elif prefix_search:
        similar_words(token)
    else:
        tf_idf_helper(token)


def check_valid_token(token):
    # @todo: Make it better
    return not re.search('[<>/]+|[0-9]+', token) and not token in punctuation and \
           not token in stop_words and token != "``" and token != "amp" and token != ".."


def cosine_sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


def create_doc_vector():
    global document_vector, vocab
    for token in vocab:
        tf_idf_helper(token)
    for doc, value in tf_idf:
        idx1 = doc_list.index(doc)
        for token, score in value:
            try:
                document_vector[idx1][vocab.index(token)] = score
            except:
                pass


def retrieve_and_rank():
    global doc_query_score, args
    top = args.cutoff
    last_tag = "</>"
    f = open(args.output, "w")
    for i, data in enumerate(query_data):
        data = data.translate(str.maketrans('', '', '[]{}(),;:+*\"?$`.'))
        data = data.translate(str.maketrans('-', ' '))
        data = re.sub(
            '(</PERSON>\s*<PERSON>)|(</LOCATION>\s*<LOCATION>)|(</ORGANI[SZ]ATION>\s*<ORGANI[SZ]ATION>)',
            '', data)
        doc_query_score = {}
        last_tag = "</>"
        token_flag = False
        for token in data.split():
            if len(token) <= 1:
                continue
            if token[0] == '<':
                last_tag = token
                if token[1] != '/':
                    token_flag = True
                else:
                    token_flag = False
            token = token.strip()
            named_entity_search = False
            prefix_search = False
            named_entity = ""
            if check_valid_token(token):
                # Prefix searches are not sent with named tags. Eg: L:New* will be sent as new and not new_L.
                # N:<word> will be sent as <word>
                if token[-1] == "*":
                    prefix_search = True
                    token = token[:-1]
                elif token[1] == ':' and token[0] in ['N', 'P', 'O', 'L']:
                    named_entity_search = True
                    named_entity = token[0]
                    token = token[2:]
                    if named_entity != 'N' and not prefix_search:  # Appendable case eg L:New
                        token = token.lower() + "_" + named_entity
                    else:
                        token = token.lower()
                else:
                    token = token.lower()
                    if token_flag:
                        token += "_" + last_tag[1]
                calculate_tf_idf(token, named_entity_search, prefix_search, named_entity)

        # This will calculate the sorted scores of query based on Matching Score Ranking
        sorted_doc_query_score = {k: v for k, v in
                                  sorted(doc_query_score.items(), key=lambda item: item[1], reverse=True)}
        res = "{qid} Q0 {docno} 0 {sim} prise1\n"
        k = 0

        normalise = True
        rms = 1
        if normalise:
            arr = np.array(list(sorted_doc_query_score.values()))
            rms = np.sqrt(np.sum(arr ** 2))
        print(sorted_doc_query_score)
        for doc, score in sorted_doc_query_score.items():
            if normalise:
                score = score / rms
            f.write(res.format(qid=query_keys[i], docno=doc, sim=score))
            k += 1
            if k == top:
                break
        f.write("\n\n")
        # This will generate scores based on cosine similarities

    f.close()


def load_posting_lists():
    global inv_index
    # with open(args.dict) as f:
    #     index = json.load(f)
    with open(args.index) as f:
        inv_index = json.load(f)
    # print(index)
    print(inv_index)


def parse(file):
    if os.path.isfile(file):
        with open(file) as f:
            copy = False
            q_num = ""
            q = ""
            for line in f:
                if line.startswith("<num>"):
                    q_num = line.rstrip().split()[-1]
                if line.startswith("<title>"):
                    copy = True
                    q += line.strip().split("Topic:", 1)[1]
                elif line.startswith("<desc>"):
                    copy = False
                    q = q.translate(str.maketrans('', '', '(),\"?'))
                    q = q.translate(str.maketrans('-/', '  '))
                    query[q_num] = q.strip()
                    q = ""
                    continue
                elif copy:
                    q += " " + line.strip()
            print(query)


# python vecsearch.py --query D:\Downloads\Repos\IRA1\topics.51-100 --index test_indexfile.idx --dict
# test_indexfile.dict --output test_resultfile
# --query D:\Downloads\Repos\IRA1\topics.51-100 --index indexfile.idx --dict indexfile.dict --output resultfile
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Amazing IR System')
    parser.add_argument('--query', type=str,
                        help='a file containing keyword queries, with each line corresponding to a query')
    parser.add_argument('--cutoff', type=int, nargs='?', default=10,
                        help='the number k (default 10) which specifies how many top-scoring results have to be '
                             'returned for each query')
    parser.add_argument('--output', type=str,
                        help='the output file named result file which is generated by your program, which contains the '
                             'document ids of all documents that have top-k (k as specified) highest-scores and their '
                             'scores in each line (note that the output could contain more than k documents). Results '
                             'for each query have to be separated by 2 newlines.')
    parser.add_argument('--index', type=str, nargs='?', default="indexfile",
                        help='the index file generated by invidx cons program above')
    parser.add_argument('--dict', type=str, nargs='?', default="indexfile",
                        help='the dictionary file generated by the invidx cons program above')
    args = parser.parse_args()
    query = {}
    parse(args.query)
    query_keys = np.array(list(query.keys()))
    query_data = np.array(list(query.values()))

    # @TODO: DELETE THE NEXT LINE!!!
    # set_java_env()

    start = time.time()

    # pool = mp.Pool(mp.cpu_count())
    # tag_entities()
    # pool.close()
    # pool.join()

    query_tagging_ends = time.time()

    print("query tagging time: " + str(query_tagging_ends - start))

    stop_words = stopwords.words('english')
    # stop_words_removal = np.vectorize(remove_stop_words)
    # stop_words_removal(query_data)
    print(query_data)
    stemmer = PorterStemmer()
    # index = {}
    inv_index = {}
    load_posting_lists()
    tf_idf = {}
    N_docs = inv_index["$doc_size$"]["$doc_size$"]
    # vocab = np.array(list(inv_index.keys())[1:])
    # doc_list = np.array(list(index.keys()))

    doc_query_score = {}
    retrieve_and_rank()

    retrieval_ends = time.time()

    print("query tagging time: " + str(query_tagging_ends - start))
    print("retrieval time: " + str(retrieval_ends - query_tagging_ends))
    print("total time: " + str(retrieval_ends - start))
