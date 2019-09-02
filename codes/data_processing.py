#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from random import shuffle
import math
import os

def query_tv(query_np):##2 通过自身向量的计算
	gi = np.multiply(np.eye(query_np.shape[0],dtype=float),np.dot(query_np,query_np.T))
	g = [math.exp(item) for item in np.sum(gi,axis=1)]
	g = [[item/np.sum(g)] for item in g]

	g = np.repeat(np.array(g),query_np.shape[1],axis=1)
	query_np = np.multiply(query_np,g)
	return query_np


def term2vector(w2v,terms):
	vectors =[]
	for term in terms:
		try:
			vectors.append(w2v[term])
		except:
			vectors.append(np.random.uniform(-0.25,0.25,50))
	vectors = np.array(vectors)
	return vectors

def _term2vector(query,doc):
	vectors =[]
	terms = list(set(query)|set(doc))
	for term in terms:
		try:
			vectors.append(w2v[term])
		except:
			vectors.append(np.random.uniform(-0.25,0.25,50))
			print term

	query_vectors =[]
	for word in query:
		query_vectors.append(vectors[terms.index(word)])
	doc_vectors =[]
	for word in doc:
		doc_vectors.append(vectors[terms.index(word)])
	return np.array(query_vectors),np.array(doc_vectors)
	

def lc_bm25(qid_docno_score_dict, prediction_data_bm25, bm25weighting, topk=1000):
	res_dict = {'questions': []}
	for qid ,value in qid_docno_score_dict.items():
		docnos = value['docno']
		scores = value['y_pred']#shape(1,1000,1)
		scores = np.array(scores)[0,:,0]
		bm25_scores = prediction_data_bm25[qid]['bm25']
		scores = (bm25weighting*np.array(bm25_scores)+scores).tolist()
		retr_scores = list(zip(docnos, scores))
		shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
		sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
		res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
	return res_dict

def lc_bm25_2(qid_docno_score_dict, prediction_data_bm25, idx, bm25weighting, idx_2, topk=1000):
	res_dict = {'questions': []}
	for qid ,value in qid_docno_score_dict.items():
		docnos = value['docno']
		scores = value['y_pred']#1*docnos*1*weights*weights
		scores = np.array(scores[0])
		scores = scores[:,0,idx,idx_2]
		bm25_scores = prediction_data_bm25[qid]['bm25']
		scores = (bm25weighting*np.array(bm25_scores)+scores).tolist()
		retr_scores = list(zip(docnos, scores))
		shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
		sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
		res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
	return res_dict


def Rank_topK(qid_docno_score_dict, idx, topk=1000):
	res_dict = {'questions': []}
	for qid ,value in qid_docno_score_dict.items():
		docnos = value['docno']
		scores = value['y_pred']
		scores = np.array(scores[0])
		scores = scores[:,0,idx]
		retr_scores = list(zip(docnos, scores))
		shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
		sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
		# try:
			# sorted_retr_scores = sorted_retr_scores[:topk]
		# except:
			# sorted_retr_scores = sorted_retr_scores
		res_dict['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
	return res_dict







