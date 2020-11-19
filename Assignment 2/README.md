# COL764 - Assignment 2: Document reranking task

## Description
In this assignment the goal is to develop “telescoping” models aimed at improving the precision of results using
pseudo-relevance feedback. The dataset we will work with is based on a recently released large collection from
Microsoft Bing called MS-MARCO. Although it was originally intended for benchmarking reading comprehension
task, it was adapted by TREC for document retrieval task.

## Topics Covered
- Analysing Dataset
- Pre-processing data.
- Pre-processing and tagging query data
- Probabilistic Retrieval Ranking
- Relevance Model based Language Modelling using Unigram Model with Dirichlet Smoothing
- Relevance Model based Language Modelling using Bigram Models with Dirichlet Smoothing with Unigram Backoff


## Running the code
- `prob_rerank.py` - Reads the input files, reranks the relevant documents based on Probabilistic retrieval methods using query expansion  
Do `python prob_rerank.py` with the following flags - 
```bash
usage: prob_rerank.py [-h] [--query-file QUERY_FILE] [--top-100-file TOP_100_FILE] [--collection-file COLLECTION_FILE]
                      [--expansion-limit EXPANSION_LIMIT]

Amazing IR System

optional arguments:
  -h, --help            show this help message and exit
  --query-file QUERY_FILE
                        file containing the queries in the same tsv format as given in Table 1 for queries file
  --top-100-file TOP_100_FILE
                        a file containing the top100 documents in the same format as train and dev top100 files given,
                        which need to be reranked
  --collection-file COLLECTION_FILE
                        file containing the full document collection (in the same format as msmarcodocs file given
  --expansion-limit EXPANSION_LIMIT
                        a number ranging from 1—15 that specifies the limit on the number of additional terms in the
                        expanded query
```
Note - this file generates the output file `prob_rerank_{expansion_limit}_{timestamp}` in the same directory as the code python file

- `lm_rerank.py` - 
Reads the input files, reranks the relevant documents based on Relevance Model based Language Modeling methods using Unigram/Bigram models with Unigram backoff and Dirichlet Smoothing  
Do `python lm_rerank.py` with the following flags - 
```bash
usage: lm_rerank.py [-h] [--query-file QUERY_FILE] [--top-100-file TOP_100_FILE] [--collection-file COLLECTION_FILE]
                    [--model MODEL]

Amazing IR System

optional arguments:
  -h, --help            show this help message and exit
  --query-file QUERY_FILE
                        file containing the queries in the same tsv format as given in Table 1 for queries file
  --top-100-file TOP_100_FILE
                        a file containing the top100 documents in the same format as train and dev top100 files given,
                        which need to be reranked
  --collection-file COLLECTION_FILE
                        file containing the full document collection (in the same format as msmarcodocs file given
  --model MODEL         it specifies the unigram or the bigram language model that should be used for relevance
                        language model.
```
Note - this file generates the output file `lm_rerank_{model}_{timestamp}` in the same directory as the code python file

- `TREC-EVAL:` Run the following command with the output file to get the NDCG scores using the TREC tool  
`./trec_eval -m ndcg -M 1 msmarco-doctrain-qrels.tsv msmarco-doctrain-top100`  
Replace all paths with the absolute paths of the corresponding files in the above command