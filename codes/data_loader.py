#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pickle
import sys, re, os
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()

def load_qrets(target):
	Dict ={}
	count = 0
	with open(target) as fin:
		for line in fin.readlines():
			qid,_,docno,rank,score,method = line.strip().split(' ')
			if qid not in Dict:
				Dict[qid] ={'docno':[],'rank':[],'bm25':[]}
			Dict[qid]['docno'].append(docno)
			Dict[qid]['rank'].append(int(rank))
			Dict[qid]['bm25'].append(float(score))
			count+=1
	print 'processing %s: %d queries %d documents'%(target,len(Dict),count)
	return Dict

def load_qrels(file):
	qrel_data ={}
	with open (file) as fin:
		for line in fin.readlines():
			qid,_,docno,tag =line.strip().split(' ')
			if tag !='0':
				if qid not in qrel_data:
					qrel_data[qid] ={'docno':[]}
				qrel_data[qid]['docno'].append(docno)
	return qrel_data

def load_w2v(w2v_file):
	if w2v_file.find('.bin')!= -1:
		w2v = KeyedVectors.load_word2vec_format(w2v_file, binary = True)
	else:
		w2v = KeyedVectors.load_word2vec_format(w2v_file, binary = False)
	print ("num words already in word2vec: " + str(len(w2v.vocab.keys())))
	return w2v

def load_sw():
	Stopwords = list(set(stopwords.words('english'))-set(['ma','most','against','t','j','e','non']))
	return Stopwords