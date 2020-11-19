import argparse
import time
import sys
import numpy as np
import pandas as pd
import csv
import re
from itertools import islice
from nltk.corpus import stopwords


def read_query_file():
    global query_with_id
    query_with_id = pd.read_csv(args.query_file, sep='\t', header=None, index_col=0).to_dict()[1]


def read_top_100_file_and_reweight():
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
            calc_offer_weights()
            # print(inv_index)
            # print(offer_weights)
            expand_query(query_tokens, args.expansion_limit)
            # print(new_doc_rank)
            write_results(query_id)

            # end_reranking = time.time()
            # print(lines_gen)
            # print("Prob Reranking time: " + str(end_reranking - start))


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
            # print("***"+res.format(qid=query_id, docno=doc, score=score))
            outfile.write(res.format(qid=query_id, docno=doc, score=score))


def expand_query(query, lim):
    for token in list(dict(sorted(offer_weights.items(), key=lambda x: x[1], reverse=True)[:lim]).keys()):
        if token not in query:
            query.append(token)
    reorder_documents(query)


def reorder_documents(query):
    avg_doc_len = inv_index["$doc_len$"]["$avg_doc_len$"]
    for doc, doc_len in inv_index["$doc_len$"].items():
        if doc == "$avg_doc_len$":
            continue
        new_doc_score = 0
        for token in query:
            if len(token) <= 1:
                continue
            token = token.lower().strip()
            if not check_valid_token(token):
                continue
            if token not in inv_index:
                continue
            n = len(inv_index[token])
            if doc in inv_index[token]:
                new_doc_score += ((np.log(N_Collection) - np.log(n)) * inv_index[token][doc] * (bm25_k1 + 1)) / (
                        bm25_k1 * ((1 - bm25_b) + (bm25_b * inv_index["$doc_len$"][doc] / avg_doc_len)) +
                        inv_index[token][doc])
        new_doc_rank[doc] = new_doc_score


def check_valid_token(token):
    # @todo: Make it better
    return token not in cachedStopWords and not re.search('[0-9]+', token)


def calc_offer_weights():
    global inv_index, word_scores
    R = 100
    N = N_Collection
    agg_len = 0
    if "$doc_len$" in inv_index:
        for doc, l in inv_index["$doc_len$"].items():
            agg_len += l
        # TODO - make 100 a variable size?
        inv_index["$doc_len$"]["$avg_doc_len$"] = agg_len / 100  # len(inv_index["$doc_len$"])
    else:
        # TODO -Is the maxsize required on the full dataset?
        inv_index["$doc_len$"] = {"$avg_doc_len$": sys.maxsize}
    for token in inv_index.keys():
        if token == "$doc_len$":
            continue
        r = len(inv_index[token])
        # TODO - make 100 a variable size?
        n = len(inv_index[token]) * N_Collection / 100  # len(inv_index["$doc_len$"])
        offer_weights[token] = r * np.log(((r - 0.5) * (N - n - R + r + 0.5)) / ((n - r + 0.5) * (R - r + 0.5)))


def load_docs():
    global inv_index, N_Collection, first_run
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
            if not first_run and count == 100:
                break
            if first_run:
                N_Collection += 1
        # if "$doc_len$" not in inv_index:
        #     inv_index["$doc_len$"] = {"$avg_doc_len$": (agg_len_doc / len(query_to_top_docs))}
        # elif inv_index["$doc_len$"]["$avg_doc_len$"] == 0.0:
        #     inv_index["$doc_len$"]["$avg_doc_len$"] = agg_len_doc / len(query_to_top_docs)  # 100
        first_run = False


# --query-file data\msmarco-docdev-queries.tsv --top-100-file data\msmarco-docdev-top100 --collection-file data\docs.tsv --expansion-limit 10
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
    parser.add_argument('--expansion-limit', type=int, dest='expansion_limit',
                        help='a number ranging from 1—15 that specifies the limit on the number of additional terms '
                             'in the expanded query')
    args = parser.parse_args()

    # initialise
    first_run = True
    N_Collection = 0
    query_to_top_docs = []
    query_with_id = {}
    inv_index = {}
    top_k = 100
    bm25_k1 = 2
    bm25_b = 0.75
    cachedStopWords = stopwords.words("english")
    offer_weights = {}
    new_doc_rank = {}
    result_file = "prob_rerank_" + str(args.expansion_limit) + "_" + str(time.strftime("%H-%M-%S", time.gmtime()))

    start = time.time()

    read_query_file()
    read_top_100_file_and_reweight()

    end_reranking = time.time()

    print("Prob Reranking time: " + str(end_reranking - start))
