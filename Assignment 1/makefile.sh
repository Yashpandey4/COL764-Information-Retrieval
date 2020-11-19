#!/bin/bash

# prepare the environment
pip install nltk
pip install numpy


# Generate the posting lists
python invidx_cons.py ./input indexfile

# Gunzip the posting lists for compact storage
tar -cvzf postings.tar.gz indexfile.*

# unzip the zipped files
tar -xvzf postings.tar.gz -C .

# print the files in human readable format
python printdict.py indexfile.dict

# retrieve the relevant documents based on query and rank
wget http://www.java2s.com/Code/JarDownload/stanford/stanford-ner.jar.zip
wget https://github.com/wolfgangmm/exist-stanford-ner/raw/master/resources/classifiers/english.all.3class.distsim.crf.ser.gz
python vecsearch.py --query ./topics.51-100 --index ./indexfile.idx --dict ./indexfile.dict --output ./resultfile --cutoff 10

# now check the TREC eval rankings. Make sure you move result file into trec_eval/results folder, along with the qrels file
cp resultfile ./trec_eval/results
cp qrels.51-100 ./trec_eval/results
cd trec_eval
./trec_eval -m set_F -M100 -m ndcg_cut.10 results/qrels.51-100 results/resultfile
