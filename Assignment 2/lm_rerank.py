import argparse
import time
import sys
import numpy as np
import pandas as pd
import csv
import re
from itertools import islice
from nltk.corpus import stopwords


maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def read_query_file():
    global query_with_id
    query_with_id = pd.read_csv(args.query_file, sep='\t', header=None, index_col=0).to_dict()[1]


def read_top_100_file_and_reweight_unigram():
    global query_to_top_docs
    with open(args.top_100_file, 'r') as file_in:
        while True:
            start = time.time()
            lines_gen = list(islice(file_in, top_k))
            if not lines_gen:
                break

            query_to_top_docs = [out_line.strip().split()[2] for out_line in lines_gen]
            query_id = lines_gen[0].split()[0]
            query_tokens = query_with_id[int(query_id)].strip().split()
            query_tokens = [word.lower().strip() for word in query_tokens]
            query_tokens = [word for word in query_tokens if check_valid_token(word)]

            load_docs()
            # print(query_to_top_docs)
            calc_scores(query_tokens)
            # print(inv_index)
            # print(new_doc_rank)
            write_results(query_id)

            # end_reranking = time.time()
            # print(lines_gen)
            # print("Prob Reranking time: " + str(end_reranking - start))


def read_top_100_file_and_reweight_bigram():
    global query_to_top_docs
    with open(args.top_100_file, 'r') as file_in:
        while True:
            start = time.time()
            lines_gen = list(islice(file_in, top_k))
            if not lines_gen:
                break

            query_to_top_docs = [out_line.strip().split()[2] for out_line in lines_gen]
            query_id = lines_gen[0].split()[0]
            query_tokens = query_with_id[int(query_id)].strip().split()
            query_tokens = [word.lower().strip() for word in query_tokens]
            query_tokens = [word for word in query_tokens if check_valid_token(word)]
            query_tokens.insert(0, "<BOD>")
            query_tokens.append("<EOD>")

            load_docs_bigram()
            # print(query_to_top_docs)
            # print(pos_index)
            calc_scores_bigram(query_tokens)
            # print(inv_index)
            # print(new_doc_rank)
            # print()
            write_results(query_id)


def write_results(query_id):
    res = "{qid} Q0 {docno} 0 {score} IndriQueryLikelihood\n"
    sorted_new_doc_rank = {k: v for k, v in
                           sorted(new_doc_rank.items(), key=lambda item: item[1], reverse=True)}
    normalise = False
    rms = 1
    if normalise:
        arr = np.array(list(sorted_new_doc_rank.values()))
        rms = np.sqrt(np.sum(arr ** 2))
    with open(result_file, "a") as outfile:
        for doc, score in sorted_new_doc_rank.items():
            if normalise:
                score = score / rms
            print("***" + res.format(qid=query_id, docno=doc, score=score))
            outfile.write(res.format(qid=query_id, docno=doc, score=score))


def prob_term_given_doc(token, doc):
    p_c_t = 0
    for doc, freq in inv_index[token].items():
        p_c_t += freq
    p_c_t /= 100
    return (inv_index[token][doc] + ds_mu * p_c_t) / (inv_index["$doc_len$"][doc] + ds_mu)


def prob_doc_given_word(doc, token):
    return prob_term_given_doc(token, doc) * prob_word(token) * 100


def prob_word(token):
    p_w = 0
    for document in inv_index[token].keys():
        p_w += prob_term_given_doc(token, document)
    p_w /= 100
    return p_w


def prob_word_given_relevance(token, query_tokens):
    p_w = prob_word(token)
    p_prod = 1
    for q_token in query_tokens:
        p_sum = 0
        for doc in inv_index["$doc_len$"].keys():
            if doc == "$avg_doc_len$":
                continue
            if token not in inv_index or q_token not in inv_index:
                continue
            if doc in inv_index[token] and doc in inv_index[q_token]:
                p_sum += prob_doc_given_word(doc, token) * prob_term_given_doc(q_token, doc)
        if p_sum != 0:
            p_prod *= p_sum
    return p_w * p_prod


def calc_scores_bigram(query_tokens):
    global inv_index, ds_mu, pos_index
    agg_len = 0
    if "$doc_len$" in inv_index:
        for doc, l in inv_index["$doc_len$"].items():
            agg_len += l
        # TODO - make 100 a variable size?
        inv_index["$doc_len$"]["$avg_doc_len$"] = agg_len / 100  # len(inv_index["$doc_len$"])
    else:
        # TODO -Is the maxsize required on the full dataset?
        inv_index["$doc_len$"] = {"$avg_doc_len$": sys.maxsize}
    ds_mu = inv_index["$doc_len$"]["$avg_doc_len$"]

    for doc, doc_len in inv_index["$doc_len$"].items():
        if doc == "$avg_doc_len$":
            continue
        new_doc_score = 0
        for i in range(len(query_tokens) - 1):
            w_curr, w_next = query_tokens[i], query_tokens[i + 1]
            if w_curr not in pos_index[doc]:
                continue
            f_w_next_curr = 0 if w_next not in pos_index[doc] else find_co_occurrences(doc, w_curr, w_next)
            jm_lambda = ds_mu / (ds_mu + inv_index["$doc_len$"][doc])
            p_bigram = jm_lambda * f_w_next_curr / inv_index[w_curr][doc]
            p_unigram = (1 - jm_lambda) * (
                    (jm_lambda * inv_index[w_curr][doc] / inv_index["$doc_len$"][doc]) + (1 - jm_lambda) * (
                    inv_index[w_curr][doc] / 100))
            new_doc_score += np.log(p_bigram + p_unigram)
        new_doc_rank[doc] = new_doc_score


def find_co_occurrences(doc, w_curr, w_next):
    global pos_index
    curr_pos_list = pos_index[doc][w_curr]
    next_pos_list = pos_index[doc][w_next]
    matches = 0
    for i in next_pos_list:
        if i - 1 in curr_pos_list:
            matches += 1
    if matches == 0:
        matches = uni_backoff_alpha * prob_word(w_next)
    return matches


def calc_scores(query_tokens):
    global inv_index, ds_mu
    agg_len = 0
    if "$doc_len$" in inv_index:
        for doc, l in inv_index["$doc_len$"].items():
            agg_len += l
        # TODO - make 100 a variable size?
        inv_index["$doc_len$"]["$avg_doc_len$"] = agg_len / 100  # len(inv_index["$doc_len$"])
    else:
        # TODO -Is the maxsize required on the full dataset?
        inv_index["$doc_len$"] = {"$avg_doc_len$": sys.maxsize}
    ds_mu = inv_index["$doc_len$"]["$avg_doc_len$"]

    for doc, doc_len in inv_index["$doc_len$"].items():
        if doc == "$avg_doc_len$":
            continue
        new_doc_score = 0
        for token in inv_index.keys():
            if token == "$doc_len$":
                continue
            p_t_m = prob_term_given_doc(token, doc)
            p_w_r = prob_word_given_relevance(token, query_tokens)
            new_doc_score += p_w_r * np.log(p_t_m)
        new_doc_rank[doc] = new_doc_score


def check_valid_token(token):
    # @todo: Make it better
    return token not in cachedStopWords and not re.search('[0-9]+', token)


def load_docs_bigram():
    global inv_index, pos_index
    with open(args.collection_file, 'r', encoding="utf8") as f:
        r = csv.reader(f)
        count = 0
        agg_len_doc = 0
        for row in r:
            row = row[0].strip().split("\t")
            # print(row[0])
            doc_id = row[0]
            if doc_id in query_to_top_docs:
                count += 1
                data = ' '.join(row[2:])
                data = data.translate(str.maketrans('', '', '[]{}()<>,;:+*^_\\\"\'?$&%#@~`.'))
                # data = data.translate(str.maketrans('', '', string.digits))
                data = data.translate(str.maketrans('-', ' '))
                data = data.split()
                data.insert(0, "<BOD>")
                data.append("<EOD>")
                print(data)
                valid_token_count = 0
                for token in data:
                    if token != "<BOD>" and token != "<EOD>":
                        if len(token) <= 1:
                            continue
                        token = token.lower().strip()
                        if not check_valid_token(token):
                            continue
                    if doc_id not in pos_index:
                        pos_index[doc_id] = {token: [valid_token_count]}
                    else:
                        pos_index[doc_id].setdefault(token, []).append(valid_token_count)
                    if token not in inv_index:
                        inv_index[token] = {doc_id: 1}
                    else:
                        inv_index[token][doc_id] = inv_index[token].get(doc_id, 0) + 1
                    valid_token_count += 1  # valid_token_count
                if "$doc_len$" not in inv_index:
                    inv_index["$doc_len$"] = {doc_id: len(data)}
                else:
                    inv_index["$doc_len$"][doc_id] = len(data)
                agg_len_doc += len(data)  # valid_token_count
            if count == 100:
                break


def load_docs():
    global inv_index
    with open(args.collection_file, 'r', encoding="utf8") as f:
        r = csv.reader(f)
        count = 0
        agg_len_doc = 0
        for row in r:
            row = row[0].strip().split("\t")
            # print(row[0])
            doc_id = row[0]
            if doc_id in query_to_top_docs:
                count += 1
                data = ' '.join(row[2:])
                data = data.translate(str.maketrans('', '', '[]{}()<>,;:+*^_\\\"\'?$&%#@~`.'))
                # data = data.translate(str.maketrans('', '', string.digits))
                data = data.translate(str.maketrans('-', ' '))
                data = data.split()
                print(data)
                # valid_token_count = 0
                for token in data:
                    if len(token) <= 1:
                        continue
                    token = token.lower().strip()
                    if not check_valid_token(token):
                        continue
                    if token not in inv_index:
                        inv_index[token] = {doc_id: 1}
                    else:
                        inv_index[token][doc_id] = inv_index[token].get(doc_id, 0) + 1
                    # valid_token_count += 1
                if "$doc_len$" not in inv_index:
                    inv_index["$doc_len$"] = {doc_id: len(data)}  # valid_token_count
                else:
                    inv_index["$doc_len$"][doc_id] = len(data)
                agg_len_doc += len(data)  # valid_token_count
            if count == 100:
                break


# --query-file data\msmarco-docdev-queries.tsv --top-100-file data\msmarco-docdev-top100 --collection-file data\docs.tsv --model uni
# --query-file data/msmarco-docdev-queries.tsv --top-100-file data/msmarco-docdev-top100 --collection-file data/docs.tsv --model uni
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Amazing IR System')
    parser.add_argument('--query-file', type=str, dest='query_file',
                        help='file containing the queries in the same tsv format as given in Table 1 for queries file')
    parser.add_argument('--top-100-file', type=str, dest='top_100_file',
                        help='a file containing the top100 documents in the same format as train and dev top100 files '
                             'given, which need to be reranked')
    parser.add_argument('--collection-file', type=str, dest='collection_file',
                        help='file containing the full document collection (in the same format as msmarcodocs file '
                             'given')
    parser.add_argument('--model', type=str, dest='model',
                        help='it specifies the unigram or the bigram language model that should be used for relevance '
                             'language model.')
    args = parser.parse_args()

    # initialise
    query_to_top_docs = []
    query_with_id = {}
    inv_index = {}
    pos_index = {}
    top_k = 100
    ds_mu = 0
    uni_backoff_alpha = 0.6
    cachedStopWords = stopwords.words("english")
    new_doc_rank = {}
    result_file = "lm_rerank_" + str(args.model) + "_" + str(time.strftime("%H-%M-%S", time.gmtime()))

    start = time.time()

    read_query_file()

    if args.model == "uni":
        read_top_100_file_and_reweight_unigram()
        end_reranking = time.time()
        print("Unigram model time: " + str(end_reranking - start))
    elif args.model == "bi":
        read_top_100_file_and_reweight_bigram()
        end_reranking = time.time()
        print("Bigram model time: " + str(end_reranking - start))
    else:
        print("Invalid Model Specified!")
