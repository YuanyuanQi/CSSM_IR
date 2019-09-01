#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pickle
import data_processing
import os

def rolling_max(A, window, num_max):
	'''Computes roling maximum of 2D array.
	A is the array, window is the length of the rolling window and num_max is the number of maximums to return for each window.
	The output is an array of size (D,N,num_max) where D is the number of 
	columns in A and N is the number of rows.
	'''
	shape = (A.shape[1], np.max([A.shape[0]-window+1, 1]), np.min([window, A.shape[0]]))
	strides = (A.strides[-1],) + (A.strides[-1]*A.shape[1],) + (A.strides[-1]*A.shape[1],)
	b = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)
	return np.sort(b, axis=2)[:,:,::-1][:,:,:num_max]

def score(query_path, qid, doc_path, docno, wind, alpha):
	Stopwords = data_loader.load_sw()
	qry_list = pickle.load(open(os.path.join(query_path, qid)))
	qry_list = list(set(qry_list)-set(Stopwords))
	query_np = data_processing.term2vector(qry_list)

	sents = pickle.load(open(os.path.join(doc_path, docno)))
	words = sum(sents, [])
	doc_np = data_processing.term2vector(words)

	#global co 
	co = len(set(qry_list)&set(words))#统计查询里有多少个单词出现在文档里
	
	Doc_Score =[]
	try:
		query_np = data_processing.query_tv(query_np)#term vectors weighting
		qd_cos = doc_np.dot(query_np.T)/np.outer(np.linalg.norm(doc_np, axis=1),np.linalg.norm(query_np, axis=1))#doc_len*query_len
		
		if wind > doc_np.shape[0]:
			length = doc_np.shape[0]
		else:
			length = wind
		Top_K = int(math.log(length))+1
		Con_Maxs = rolling_max(temp, length, Top_K)#[query_len*(doc_len-45+1)*2] 
		Con_Score = np.sum(Con_Maxs, axis=0)#[(doc_len-45+1)*2]
		score_al = np.max(Con_Score[:,0] + alpha*np.mean(Con_Score,axis=1))*math.log(co+args.C)
		Doc_Score.append([score_al.tolist()])
	except:
		# Doc_Score.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		Doc_Score.append([[0.0]])
	return Doc_Score