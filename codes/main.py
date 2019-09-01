#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import pickle
import sys, re, os
import multiprocessing
import numpy as np
from collections import OrderedDict
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
from collections import defaultdict
from random import shuffle

import data_loader
import lf
import cf
import evaluation
import data_processing




def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()

#system settings
parser = argparse.ArgumentParser()

args = parser.parse_args()
parser.add_argument('--target', type=str, default="wt2g", help='what trec dataset to retrieval: wt2g,ap8889,robust04,wt10g,blog06')
parser.add_argument('--function', type=str, default="linearly function", help='what function to score: linearly function or co-weighted function')
parser.add_argument('--window_width', type=int, default=35, help='text window width for documents')
parser.add_argument('--alpha', type=float, default=0.1, help='infulence factor to balance semantic interactions weighting')
parser.add_argument('--beta', type=float, default=0.05, help='infulence factor to balance BM25 method weighting')
parser.add_argument('--C', type=int, default=20, help='constant to balance parameter co')
args = parser.parse_args()

print '************** Loading Data**************'

data_directory = '../data'
w2v_file = data_directory+'/vector_bins/glove.6B.50d.g.txt'
# global query_path,doc_path,qrels_path,qrets_path
query_path = data_directory+'/queries/'+args.target
doc_path = data_directory+'/'+args.target+'/docs/'

qrels_path = data_directory+'/qrels/qrels.'+args.target
qrets_path = data_directory+'/qrets/'+args.target+'.res'


w2v = data_loader.load_w2v(w2v_file)

# prediction_data = load_qrets(qrets_path)
# golden_data = load_qrels(qrels_path)

print '*****************Scoring **************'
#global wind 
wind = args.window_width

prediction_data = data_loader.load_qrets(qrets_path)
prediction_score ={}
count=0
if args.function == 'linearly function':
	print args.function
	for qid,value in prediction_data.items()[:1]:
		count += 1
		print 'processing qid %s: %d / %d'%(qid,count,len(prediction_data))
		y_pred =[]
		for docno in value['docno']:
			doc_scores = lf.score(query_path, qid, doc_path, docno, wind, args.alpha)# windows+top2+weights
			y_pred.append(doc_scores)
		if qid not in prediction_score:
			prediction_score[qid] = {'docno':[],'y_pred':[]}
		prediction_score[qid]['docno']=value['docno']
		prediction_score[qid]['y_pred'].append(y_pred)
else:
	for qid,value in prediction_data.items():
		count += 1
		print 'processing qid %s: %d / %d'%(qid,count,len(prediction_data))
		y_pred =[]
		for docno in value['docno']:
			doc_scores = cw.score(query_path, qid, doc_path, docno, wind, args.alpha)# windows+top2+weights
			y_pred.append(doc_scores)
		if qid not in prediction_score:
			prediction_score[qid] = {'docno':[],'y_pred':[]}
		prediction_score[qid]['docno']=value['docno']
		prediction_score[qid]['y_pred'].append(y_pred)


# qid_doc_score = qid_docs_scores(qrets_path, wind, alpha)

print '************** Combining with BM25 *************************'




score_bm25 = data_loader.load_qrets(qrets_path)
golden_data = data_loader.load_qrels(qrels_path)
res_dict = data_processing.lc_bm25(prediction_score, score_bm25, 0, args.beta, topk=1000)

res_dict_bm25 = {'questions': []}
for qid ,value in score_bm25.items():
	docnos = value['docno']
	bm25_scores = score_bm25[qid]['bm25']
	retr_scores = list(zip(docnos, bm25_scores))
	shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
	sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
	res_dict_bm25['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})

print '************** BM25 Evaluation  **************'
evaluation.trec(golden_data, res_dict_bm25, data_directory+'/results/'+args.target+'_bm25_0.35.eval')
print '************** Combining BM25 Evaluation **************'
evalfile_name = data_directory+'/results/'+args.target+'_win{0}_alpha{1}_bm25_beta{2}.eval'.format(wind, args.alpha, args.beta)
evaluation.trec(golden_data, res_dict, evalfile_name)