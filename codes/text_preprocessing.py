# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:22:28 2018

@author: qiyy
# """
import io, os, sys
import gzip 
import pickle
import re
import codecs
import nltk  
import nltk.data 
import random
import math
import string
import operator
import multiprocessing
import numpy as np
from nltk.stem.porter import * 
from nltk.corpus import stopwords
from compiler.ast import flatten 
import matplotlib.pyplot as plt
from collections import Counter
import time
from krovetzstemmer import Stemmer

# clean = lambda t: re.sub('[,?;*!%^&_+():-\[\]{}]', ' ', t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'", ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('-', ' ').replace('.', '').replace('&hyph;', ' ').replace('&blank;', ' ').strip().lower())

# clean = lambda t: re.sub('[,?;*!%^&_+():-\[\]{}]', ' ', t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'", ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('-', ' ').replace('.', '').replace('&hyph;', ' ').replace('&blank;', ' ').strip().lower())
clean = lambda t: re.sub('[^&\.a-z0-9]', ' ', t.strip())
UNK_TOKEN = '*UNK*'


def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()

def stemm(index):
	print 'Sub_Processing : '+str(index)
	Kicks = set(kick_tf)|set(kick_idf)
	for docno in fold_files[IDX*index : IDX*(index+1)]:
		if docno in os.listdir(Locdir+'/MQ0708_kstem_dict/'):
			D = pickle.load(open(Locdir+'/MQ0708_kstem_dict/'+docno))
			new_value =[]
			sents = D.values()[0]
			# saveObjToFile(Locdir+'/docs_stem/'+docno,sents)
			for sent in sents:
				if len(sent) >1:
					kick = set(sent)&set(Kicks)
					if len(kick)>0:
						for word in kick:
							sent.remove(word)
						if len(sent)>1:
							new_value.append(sent)
					else:
						new_value.append(sent)
			saveObjToFile(Locdir+'/docs_stem_kick/'+docno,new_value)
		else:
			print docno,' not found in MQ0708_kstem_dict'

def AP8889_stem():
	Docnos = []
	files = ['/AP8889/DATASETS/Disk1/AP89','/AP8889/DATASETS/Disk2/AP88/']
	# tags: DOCNO TEXT
	for file in files:
		print 'Processing: ',file
		if os.path.isdir(Locdir+"/AP8889/docs_stem/"):
			pass
		else:
			os.mkdir(Locdir+"/AP8889/docs_stem/")
		for file_name in os.listdir(Locdir+file):
			# if file_name == 'AP890906.gz':
			f_name = file_name.replace(".gz", "")
			#获取文件的名称，去掉
			g_file = gzip.GzipFile(Locdir+file+'/'+file_name)
			#创建gzip对象
			Text = g_file.read()
			# DOCNOs = re.findall(r'<DOCNO>(.*?)</DOCNO>',Text,re.S)
			Texts = re.findall(r'<DOC>(.*?)</DOC>',Text,re.S)
			for Text in Texts:
				docno = re.findall(r'<DOCNO>(.*?)</DOCNO>',Text,re.S)[0]
				Docnos.append(str(docno.strip()))
				''' Text = re.findall(r'<TEXT(.+?)</TEXT>',Text,re.S)[0]
					# Text = re.sub(r'<.*?>|&nbsp;|&amp;|\||\(|\)|\/|&#|','',Text.replace("\r"," ").replace('\t',' ').replace('\n',' '))
					# Text = Text.replace("'"," '")
					# sents = nltk.sent_tokenize(Text)
					# sents_new =[]
					# for sent in sents:
						# sent = re.sub(r'[^a-z0-9.\'-]',' ',sent.lower())
						# words = nltk.word_tokenize(sent)
						# words =[' '.join(re.findall(r'\'?[a-z0-9]+\'?\-?\.?[a-z0-9]*\.?[a-z0-9]*',word)) for word in words if re.findall(r'[a-z0-9]',word) != []]
						# words = [krovetzstemmer.stem(word) for word in words]
						# sents_new.append(words)
					# saveObjToFile(Locdir+"/AP8889/docs_stem/"+docno,sents_new)#'''

def AP8889_kick():
	if os.path.exists(Locdir+'/AP8889/ap8889.kick.stem'):
		print 'Loading kick words'
		kick = pickle.load(open(Locdir+'/AP8889/ap8889.kick.stem'))
		words_alpha_stem = pickle.load(open(Locdir+'/words.alpha.stemmed'))
		# print set(kick)&set(words_alpha_stem),len(set(kick)&set(words_alpha_stem)),len(kick)
		kicks = list(set(kick)-set(words_alpha_stem))
		if os.path.isdir(Locdir+"/AP8889/docs_stem_kick/"):
			pass
		else:
			os.mkdir(Locdir+'/AP8889/docs_stem_kick/')
		for docno in os.listdir(Locdir+'/AP8889/docs_stem/'):
			sents = pickle.load(open(Locdir+'/AP8889/docs_stem/'+docno))
			new_sents = []
			for sent in sents:
				if len(sent) >1:
					Kick = set(sent)&set(kicks)
					if len(Kick)>0:
						for word in Kick:
							sent.remove(word)
						if len(sent)>1:
							new_sents.append(sent)
					else:
						new_sents.append(sent)
			saveObjToFile(Locdir+'/AP8889/docs_stem_kick/'+docno,new_sents)
	else:
		print 'processing kick vocab'
		words_tf = []
		words_idf = []
		qid_wordslist = {}
		with open('/home/qiyy/Documents/AP8889/Topics_Qrels/topics/topics.AP8889.51-200') as fin:
			text = fin.read()
			Texts = re.findall(r'<top>(.*?)</top>',text,re.S)
			for Text in Texts:
				qid = re.findall(r'Number:(.*)',Text)[0]
				query = re.findall(r'<title>(.*?)<desc>',Text.replace('\r\n',' '),re.S)[0]
				query = query.strip().lower().replace("'"," '").replace("-"," '")
				query_words = nltk.word_tokenize(query)
				query_words = [word for word in query_words if re.findall(r'[a-z0-9]',word) != []]
				qid_wordslist[qid.strip()] = query_words
		saveObjToFile(Locdir+'/AP8889/ap8889.150.orig',qid_wordslist)
		
		for docno in os.listdir(Locdir+'/AP8889/docs_stem/'):
			sents = pickle.load(open(Locdir+'/AP8889/docs_stem/'+docno))
			words_tf.extend(sum(sents,[]))
			words_idf.extend(list(set(sum(sents,[]))))
		vocab_tf_stem = Counter(words_tf)
		del words_tf
		vocab_idf_stem = Counter(words_idf)
		del words_idf
		saveObjToFile(Locdir+'/AP8889/vocab.tf.stem.ap',vocab_tf_stem)
		saveObjToFile(Locdir+'/AP8889/vocab.idf.stem.ap',vocab_idf_stem)
		
		stopwords_stem = [krovetzstemmer.stem(word) for word in Stopwords]
		stopwords_stem.remove('against')
		qid_wordslist_stem = {}
		for qid,words in qid_wordslist.items():
			qid_wordslist_stem[qid] = [krovetzstemmer.stem(word) for word in words if krovetzstemmer.stem(word) not in stopwords_stem]
		saveObjToFile(Locdir+'/AP8889/ap8889.150.stem',qid_wordslist_stem)
		query_words_stem = [krovetzstemmer.stem(word) for word in sum(qid_wordslist.values(),[])]
		kick_idf = [word for word,tf in vocab_tf_stem.items() if tf<50]
		kick_tf = [word for word,idf in vocab_idf_stem.items() if idf<10]
		Kicks = list(set(kick_idf)|set(kick_tf)|set(stopwords_stem)-set(query_words_stem))
		print len(Kicks)
		saveObjToFile(Locdir+'/AP8889/ap8889.kick.stem',Kicks)

def Disk12_stem():
	docs_paths = ['/home/qiyy/Documents/disk1+2/data/disk2/address_disk2','/home/qiyy/Documents/disk1+2/data/disk1/address_disk1']
	#tags : DOCNO,TEXT
	if os.path.isdir(Locdir+"/disk1+2/docs_stem/"):
		pass
	else:
		os.mkdir(Locdir+"/disk1+2/docs_stem/")
	for item in docs_paths[1:]:
		with open(item) as fin:
			for line in fin.readlines():
				g_file = gzip.GzipFile(line.strip())
				#创建gzip对象
				Text = g_file.read()
				# DOCNOs = re.findall(r'<DOCNO>(.*?)</DOCNO>',Text,re.S)
				Texts = re.findall(r'<DOC>(.*?)</DOC>',Text,re.S)
				for Text in Texts:
					docno = re.findall(r'<DOCNO>(.*?)</DOCNO>',Text,re.S)[0]
					docno = docno.strip()
					# print docno
					# print Text
					Text = re.findall(r'<TEXT(.+?)</TEXT>',Text,re.S)[0]
					Text = re.sub(r'<.*?>|&nbsp;|&amp;|\||\(|\)|\/|&#|&M|&P|&o','',Text.replace("\r"," ").replace('\t',' ').replace('\n',' ').replace("'"," '"))
					Text = Text
					sents = nltk.sent_tokenize(Text)
					# print sents
					sents_new =[]
					for sent in sents:
						sent = re.sub(r'[^a-z0-9.\'-]',' ',sent.lower())
						words = nltk.word_tokenize(sent)
						words =[' '.join(re.findall(r'\'?[a-z0-9]+\'?\-?\.?[a-z0-9]*\.?[a-z0-9]*',word)) for word in words if re.findall(r'[a-z0-9]',word) != []]
						words = [krovetzstemmer.stem(word) for word in words]
						sents_new.append(words)
						# print words
					saveObjToFile(Locdir+"/disk1+2/docs_stem/"+docno,sents_new)#'''

def Disk12_kick():
	if os.path.exists(Locdir+'/disk1+2/disk12.kick.stem'):
		print 'Loading kick words'
		kick = pickle.load(open(Locdir+'/disk1+2/disk12.kick.stem'))
		words_alpha_stem = pickle.load(open(Locdir+'/words.alpha.stemmed'))
		kicks = list(set(kick)-set(words_alpha_stem))
		if os.path.isdir(Locdir+"/disk1+2/docs_stem_kick/"):
			pass
		else:
			os.mkdir(Locdir+'/disk1+2/docs_stem_kick/')
		for docno in os.listdir(Locdir+'/disk1+2/docs_stem/'):
			sents = pickle.load(open(Locdir+'/disk1+2/docs_stem/'+docno))
			new_sents = []
			for sent in sents:
				if len(sent) >1:
					Kick = set(sent)&set(kicks)
					if len(Kick)>0:
						for word in Kick:
							sent.remove(word)
						if len(sent)>1:
							new_sents.append(sent)
					else:
						new_sents.append(sent)
			saveObjToFile(Locdir+'/disk1+2/docs_stem_kick/'+docno,new_sents)
	else:
		print 'generating kick vocab'
		words_tf = []
		words_idf = []
		qid_wordslist = {}
		with open('/home/qiyy/Documents/disk1+2/topics/topics.disk1+2') as fin:
			text = fin.read()
			Texts = re.findall(r'<top>(.*?)</top>',text,re.S)
			for Text in Texts:
				qid = re.findall(r'Number:(.*)',Text)[0]
				query = re.findall(r'<title>(.*?)<desc>',Text.replace('\r\n',' '),re.S)[0]
				query = query.strip().lower().replace("'"," '").replace("-"," '").replace('topic:',' ')
				# query_words = nltk.word_tokenize(query)
				query_words = query.split()
				query_words = [word for word in query_words if re.findall(r'[a-z0-9]',word) != []]
				qid_wordslist[qid.strip()] = query_words
		saveObjToFile(Locdir+'/disk1+2/disk12.150.orig',qid_wordslist)
		
		for docno in os.listdir(Locdir+'/disk1+2/docs_stem/'):
			sents = pickle.load(open(Locdir+'/disk1+2/docs_stem/'+docno))
			words_tf.extend(sum(sents,[]))
			words_idf.extend(list(set(sum(sents,[]))))
		vocab_tf_stem = Counter(words_tf)
		del words_tf
		vocab_idf_stem = Counter(words_idf)
		del words_idf
		saveObjToFile(Locdir+'/disk1+2/vocab.tf.stem.disk12',vocab_tf_stem)
		saveObjToFile(Locdir+'/disk1+2/vocab.idf.stem.disk12',vocab_idf_stem)
		
		stopwords_stem = [krovetzstemmer.stem(word) for word in Stopwords]
		stopwords_stem.remove('against')
		saveObjToFile('stopwords.stem',stopwords_stem)
		qid_wordslist_stem = {}
		for qid,words in qid_wordslist.items():
			qid_wordslist_stem[qid] = [krovetzstemmer.stem(word) for word in words if krovetzstemmer.stem(word) not in stopwords_stem]
		saveObjToFile(Locdir+'/disk1+2/disk12.150.stem',qid_wordslist_stem)
		query_words_stem = [krovetzstemmer.stem(word) for word in sum(qid_wordslist.values(),[])]
		kick_idf = [word for word,tf in vocab_tf_stem.items() if tf<50]
		kick_tf = [word for word,idf in vocab_idf_stem.items() if idf<10]
		Kicks = list(set(kick_idf)|set(kick_tf)|set(stopwords_stem)-set(query_words_stem))
		print len(Kicks)
		saveObjToFile(Locdir+'/disk1+2/disk12.kick.stem',Kicks)#'''

def blogs06():
	qrets_files = ['/home/qiyy/Downloads/qiyy/terrier-3.5/var/results_blogs06/blogs06_BM25b0.35_0.res','/home/qiyy/Downloads/qiyy/terrier-3.5/var/results_blogs06/blogs06-08_BM25b0.35_1.res']
	stores = {}
	for qrets_file in qrets_files:
		with open (qrets_file) as fin:
			for line in fin.readlines():
				qid,_,docno,rank,score,method = line.strip().split(' ')
				tar,time,num,no = docno.split('-')
				if time+'/'+num not in stores:
					stores[time+'/'+num] = []
				stores[time+'/'+num].append(no)
	Miss =[]
	for key,nolist in stores.items():#key = time+'/'+num
		time,num = key.split('/')
		tar_path = time+'/permalinks-'+num
		pre = 'BLOG06-'+time+'-'+num+'-'
		docnos = [pre+no for no in nolist]
		count = 0
		with codecs.open(Locdir+'/big2datas/blogs06/'+tar_path,'rb','utf-8') as f:
			DOCS = re.findall(r'<DOC>.*?</DOC>',f.read(),re.S)
			for DOC in DOCS:
				docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)[0]
				docid = docid.strip()
				if docid in docnos:
					count += 1
					DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
					doc_list= []
					title = re.findall(r'<title>(.*?)</title>',DOC)
					if title != []:
						title = re.sub(r'<[^>]*?>',' ',title[0])
						title_words = nltk.word_tokenize(clean(title))
						doc_list.append(title_words)
					else:
						pass
					body = re.findall(r'<body[^>]*?>(.*?)</body>',DOC)
					if body !=[]:
						text = re.sub(r'<[^>]*?>',' ',body[0])
						sents = nltk.sent_tokenize(text)
						for sent in sents:
							words = nltk.word_tokenize(clean(sent))
							doc_list.append(words)
					else:
						Miss.append(tar_path)
					if doc_list != []:
						saveObjToFile(Locdir+'/big2datas/docs_blogs06/'+docid,doc_list)
					else:
						print 'empty doc text is  ',docid
		print 'Total %d ,only find %d'%(len(docnos),count)
	saveObjToFile('Missing_body_title_tags',list(set(Miss)))
	saveObjToFile('stores',stores)
	for key,nolist in stores.items():#key = time+'/'+num
		time,num = key.split('/')
		tar_path = time+'/permalinks-'+num
		if tar_path in Miss:
			pre = 'BLOG06-'+time+'-'+num+'-'
			docnos = [pre+no for no in nolist]
			with open(Locdir+'/big2datas/blogs06/'+tar_path,'rb') as f:
				DOCS = re.findall(r'<DOC>.*?</DOC>',f.read(),re.S)
				for DOC in DOCS:
					docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
					if docid == []:
						print tar_path,' is missing docno tag'
					else:
						DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
						title = re.findall(r'<title>(.*?)</title>',DOC)
						if title == []:
							print tar_path,' is missing title tag ',docid[0]
						body = re.findall(r'<body>(.*?)</body>',DOC)
						if body == []:
							print tar_path,' is missing body tag ',docid[0]

def blogs06_misstags():
	Miss = pickle.load(open('Missing_body_title_tags','rb'))
	stores = pickle.load(open('stores','rb'))
	Docnos =[]
	for key,nolist in stores.items():#key = time+'/'+num
		time,num = key.split('/')
		tar_path = time+'/permalinks-'+num
		if tar_path in Miss:
			pre = 'BLOG06-'+time+'-'+num+'-'
			docnos = [pre+no for no in nolist]
			Docnos.extend(docnos)
	found_docnos=  os.listdir('/home/qiyy/Documents/big2datas/docs_blogs06/')
	left_docnos = list(set(Docnos)-set(found_docnos))
	print len(left_docnos)
	for docno in left_docnos:
		tar,time,num,no = docno.split('-')
		tar_path = time+'/permalinks-'+num
		with codecs.open(Locdir+'/big2datas/blogs06/'+tar_path,'rb',encoding='utf-8',errors='ignore') as f:
				DOCS = re.findall(r'<DOC>.*?</DOC>',f.read(),re.S)
				for DOC in DOCS:
					docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
					if docid[0] in left_docnos:
						docid = docid[0].strip()
						# print docid
						DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
						doc_list= []#html 部分包括了title和body部分了
						# html = re.findall(r'<html[^>]*?>(.*?)</html>',DOC.lower())
						text = re.sub(r'<doc>.*?</dochdr>',' ',DOC.lower())
						text = re.sub(r'<[^>]*?>',' ',text)
						sents = nltk.sent_tokenize(text)
						for sent in sents:
							words = nltk.word_tokenize(clean(sent))
							doc_list.append(words)
						if doc_list != []:
							# pass
							saveObjToFile(Locdir+'/big2datas/docs_blogs06/'+docid,doc_list)
						else:
							print docid,text
	'''founds = os.listdir('/home/qiyy/Documents/big2datas/docs_blogs06/')
			docnos = list(set(docnos)-set(founds))
			with open(Locdir+'/big2datas/blogs06/'+tar_path,'rb') as f:
				DOCS = re.findall(r'<DOC>.*?</DOC>',f.read(),re.S)
				for DOC in DOCS:
					docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)[0]
					if docid in docnos:
						doc_list = []
						DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
						title = re.findall(r'<title>(.*?)</title>',DOC)
						if title != []:
							title =' '.join(title)
							title = re.sub(r'<[^>]*?>',' ',title)
							title_words = nltk.word_tokenize(clean(title))
							doc_list.append(title_words)
						body = re.findall(r'<body[^>]*?>(.*?)</body>',DOC)
						body = re.sub(r'<[^>]*?>',' ',body[0])
						
						if body == []:
							print tar_path,' is missing body tag ',docid[0]
						else:
							sents = nltk.sent_tokenize(body)
							for sent in sents:
								words = nltk.word_tokenize(clean(sent))
								doc_list.append(words)
						if doc_list != []:
							# pass
							saveObjToFile(Locdir+'/big2datas/docs_blogs06/'+docid,doc_list)
						else:
							print 'empty doc %s'%docid#'''

def WT10G():
	qrers_file = '/home/qiyy/Documents/big2datas/wt10g.BM25b0.35_0.res'
	stores = {}
	with open (qrers_file) as fin:
		for line in fin.readlines():#451 Q0 WTX064-B48-188 0 15.756777120193739 BM25b0.35
			qid,_,docno,rank,score,method = line.strip().split(' ')#51 Q0 AP880318-0287 0 15.07665404976806 BM25b0.35
			WTX,B,no = docno.split('-')
			if WTX+'/'+B not in stores:
				stores[WTX+'/'+B] = []
			stores[WTX+'/'+B].append(no)
	Miss =[]
	time = 0
	for file_path,nolist in stores.items():#key = time+'/'+num
		time += 1
		print '%d/%d'%(time,len(stores))
		WTX,B = file_path.split('/')
		pre = WTX+'-'+B+'-'
		docnos = [pre+no for no in nolist]
		count = 0
		input_file = gzip.open(Locdir+'/big2datas/WT10G/'+file_path+'.gz','rb')
		g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
		DOCS = re.findall(r'<DOC>.*?</DOC>',g_file.read(),re.S)
		for DOC in DOCS:
			docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
			if docid !=[]:
				docid = docid[0].strip()
				if docid in docnos:
					count += 1
					DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
					doc_list= []#html 部分包括了title和body部分了
					html = re.findall(r'<html[^>]*?>(.*?)</html>',DOC.lower())
					if html != []:
						text = re.sub(r'<[^>]*?>',' ',html[0])
						sents = nltk.sent_tokenize(text)
						for sent in sents:
							words = nltk.word_tokenize(clean(sent))
							doc_list.append(words)
						if doc_list != []:
							saveObjToFile(Locdir+'/big2datas/docs_wt10g/'+docid,doc_list)
						else:
							# print 'empty doc text is  ',docid
							Miss.append(docid)
					else:
						text = re.sub(r'<doc>.*?</dochdr>',' ',DOC.lower())
						text = re.sub(r'<[^>]*?>',' ',text)
						sents = nltk.sent_tokenize(text)
						for sent in sents:
							words = nltk.word_tokenize(clean(sent))
							doc_list.append(words)
						if doc_list != []:
							saveObjToFile(Locdir+'/big2datas/docs_wt10g/'+docid,doc_list)
						else:
							Miss.append(docid)
					
		print 'Total %d ,only find %d'%(len(docnos),count)
	saveObjToFile('disk12.Missing_body_title_tags',list(set(Miss)))
	saveObjToFile('disk12.stores',stores)#'''
	stores = pickle.load(open('disk12.stores','rb'))
	Docnos = []
	for file_path,nolist in stores.items():#key = time+'/'+num
		WTX,B = file_path.split('/')
		pre = WTX+'-'+B+'-'
		docnos = [pre+no for no in nolist]
		Docnos.extend(docnos)
	miss_docnos = pickle.load(open('disk12.Missing_body_title_tags','rb'))
	found_docnos = os.listdir(Locdir+'/big2datas/docs_wt10g/')
	left_docnos = list(set(Docnos)-set(found_docnos))
	print len(left_docnos)
	for docno in left_docnos:
		WTX,B,no = docno.split('-')
		file_path = WTX+'/'+B
		input_file = gzip.open(Locdir+'/big2datas/WT10G/'+file_path+'.gz','rb')
		g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
		DOCS = re.findall(r'<DOC>.*?</DOC>',g_file.read(),re.S)
		for DOC in DOCS:
			docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
			if docid !=[]:
				docid = docid[0].strip()
				if docid in left_docnos:
					DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
					doc_list= []#html 部分包括了title和body部分了
					# html = re.findall(r'<html[^>]*?>(.*?)</html>',DOC.lower())
					text = re.sub(r'<doc>.*?</dochdr>',' ',DOC.lower())
					text = re.sub(r'<[^>]*?>',' ',text)
					sents = nltk.sent_tokenize(text)
					for sent in sents:
						words = nltk.word_tokenize(clean(sent))
						doc_list.append(words)
					if doc_list != []:
						saveObjToFile(Locdir+'/big2datas/docs_wt10g/'+docid,doc_list)
					else:
						print docid,text

def disk12():
	docnos = []
	qrets_file = '/home/qiyy/Documents/disk12.BM25b0.35_0.res'
	with open(qrets_file,'rb') as fin:
		for line in fin.readlines():
			qid,_,docno,rank,score,method = line.strip().split(' ')#51 Q0 AP880318-0287 0 15.07665404976806 BM25b0.35
			docnos.append(docno)
	rootpath = '/home/qiyy/Documents/big2datas/disk1+2/data/'
	Miss =[]
	founds = os.listdir('/home/qiyy/Documents/big2datas/docs_disk12/')
	print len(founds),founds[0] #AP880318-0287
	lefts = list(set(docnos)-set(founds))
	ffs = []
	for docno in lefts:
		# print docno
		ff = re.findall(r'([A-Za-z]*?)[0-9].*?',docno)[0]
		# print ff
		ffs.append(ff)
	ffs =list(set(ffs))
	
	for first_file in os.listdir(rootpath):#disk1 disk2
		for second_file in os.listdir(rootpath+first_file):
			if second_file in ffs:
				for third_file in os.listdir(rootpath+first_file+'/'+second_file):
						if os.path.isdir(rootpath+first_file+'/'+second_file+'/'+third_file):#third_file =1990
							for gz_file in os.listdir(rootpath+first_file+'/'+second_file+'/'+third_file):
								if gz_file.find('.gz')!=-1:
									input_file = gzip.open(rootpath+first_file+'/'+second_file+'/'+third_file+'/'+gz_file, 'rb')
									g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
									Text = g_file.read()
									DOCS = re.findall(r'<DOC>.*?</DOC>', Text, re.S)
									for DOC in DOCS:
										docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
										if docid !=[]:
											doc_list = []
											docid = docid[0].strip()
											if docid in lefts:
												DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
												Text = re.findall(r'<text>(.*?)</text>',DOC)
												head = re.findall(r'<head>(.*?)</head>',DOC)
												title = re.findall(r'<title>(.*?)</title>',DOC)
												hl = re.findall(r'<hl>(.*?)</hl>',DOC)
												Title = ' '.join(head)+' '+' '.join(title)+' '+' '.join(hl)
												Title = re.sub(r'<[^>]*?>',' ',Title)
												doc_list.append(nltk.word_tokenize(clean(Title)))
												if Text != []:
													text = re.sub(r'<[^>]*?>',' ',Text[0])
													sents = nltk.sent_tokenize(text)
													for sent in sents:
														words = nltk.word_tokenize(clean(sent))
														doc_list.append(words)
												else:
													Miss.append(docid)
												if doc_list != []:
													saveObjToFile(Locdir+'/big2datas/docs_disk12/'+docid,doc_list)
													print docid
						else:##file = gz
							if third_file.find('.gz') != -1:
								input_file = gzip.open(rootpath+first_file+'/'+second_file+'/'+third_file, 'rb')
								g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
								Text = g_file.read()
								DOCS = re.findall(r'<DOC>.*?</DOC>', Text, re.S)
								for DOC in DOCS:
									docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
									if docid !=[]:
										doc_list = []
										docid = docid[0].strip()
										if docid in lefts:
											DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
											Text = re.findall(r'<text>(.*?)</text>',DOC)
											head = re.findall(r'<head>(.*?)</head>',DOC)
											title = re.findall(r'<title>(.*?)</title>',DOC)
											hl = re.findall(r'<hl>(.*?)</hl>',DOC)
											Title = ' '.join(head)+' '+' '.join(title)+' '+' '.join(hl)
											Title = re.sub(r'<[^>]*?>',' ',Title)
											doc_list.append(nltk.word_tokenize(clean(Title)))
											if Text != []:
												text = re.sub(r'<[^>]*?>',' ',Text[0])
												sents = nltk.sent_tokenize(text)
												for sent in sents:
													words = nltk.word_tokenize(clean(sent))
													doc_list.append(words)
											else:
												Miss.append(docid)
											if doc_list != []:
												saveObjToFile(Locdir+'/big2datas/docs_disk12/'+docid,doc_list)
												print docid
	print Miss
	'''time = 0
	for gz,nolist in Dict.items():
		gz = gz+'.gz'
		if gz in Path:
			time += 1
			print '%d   %d'%(time,len(Dict))
			gz_path = Path[gz]
			# Docnos = [gz+'-'+no for no in nolist]
			input_file = gzip.open(gz_path, 'rb')
			g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
			Text = g_file.read()
			DOCS = re.findall(r'<DOC>.*?</DOC>', Text, re.S)
			for DOC in DOCS:
				docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
				if docid !=[]:
					doc_list = []
					docid = docid[0].strip()
					if docid in lefts:
						DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
						Text = re.findall(r'<text>(.*?)</text>',DOC)
						head = re.findall(r'<head>(.*?)</head>',DOC)
						title = re.findall(r'<title>(.*?)</title>',DOC)
						hl = re.findall(r'<hl>(.*?)</hl>',DOC)
						Title = ' '.join(head)+' '+' '.join(title)+' '+' '.join(hl)
						Title = re.sub(r'<[^>]*?>',' ',Title)
						doc_list.append(nltk.word_tokenize(clean(Title)))
						if Text != []:
							text = re.sub(r'<[^>]*?>',' ',Text[0])
							sents = nltk.sent_tokenize(text)
							for sent in sents:
								words = nltk.word_tokenize(clean(sent))
								doc_list.append(words)
						else:
							Miss.append(docid)
						if doc_list != []:
							saveObjToFile(Locdir+'/big2datas/docs_disk12/'+docid,doc_list)
		else:
			# pass
			print gz#'''
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	'''for first_file in os.listdir(rootpath):#disk1 disk2
		for second_file in os.listdir(rootpath+first_file):#ZIFF AP
			if os.path.isdir(rootpath+first_file+'/'+second_file):
				for file in os.listdir(rootpath+first_file+'/'+second_file):#gz or 1990
					if os.path.isdir(rootpath+first_file+'/'+second_file+'/'+file):#WSJ-1990
						for gz_file in os.listdir(rootpath+first_file+'/'+second_file+'/'+file):
							if file.find('.gz') != -1:#ap8889 gz wt2g GZ
								# input_file = gzip.open(rootpath+first_file+'/'+second_file+'/'+file, 'rb')
								# g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
								# Text = g_file.read()
								pass
							else:
								fin = codecs.open(rootpath+first_file+'/'+second_file+'/'+file, 'r',encoding ='utf-8')
								Text = fin.read()
							DOCS = re.findall(r'<DOC>.*?</DOC>', Text, re.S)
							for DOC in DOCS:
								docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
								if docid !=[]:
									doc_list = []
									docid = docid[0].strip()
									if docid in docnos:
										DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
										Text = re.findall(r'<text>(.*?)</text>',DOC)
										head = re.findall(r'<head>(.*?)</head>',DOC)
										title = re.findall(r'<title>(.*?)</title>',DOC)
										hl = re.findall(r'<hl>(.*?)</hl>',DOC)
										Title = ' '.join(head)+' '+' '.join(title)+' '+' '.join(hl)
										Title = re.sub(r'<[^>]*?>',' ',Title)
										doc_list.append(nltk.word_tokenize(clean(Title)))
										if Text != []:
											text = re.sub(r'<[^>]*?>',' ',Text[0])
											sents = nltk.sent_tokenize(text)
											for sent in sents:
												words = nltk.word_tokenize(clean(sent))
												doc_list.append(words)
										else:
											Miss.append(docid)
										if doc_list != []:
												saveObjToFile(Locdir+'/big2datas/docs_disk12/'+docid,doc_list)
										else:
											Miss.append(docid)
					else:#gz
						# print 'Processing: ',rootpath+first_file+'/'+second_file+'/'+file
						if file.find('.gz') != -1:#ap8889 gz wt2g GZ
							input_file = gzip.open(rootpath+first_file+'/'+second_file+'/'+file, 'rb')
							g_file = io.TextIOWrapper(input_file, encoding='utf-8',errors='ignore')
							Text = g_file.read()
						else:
							fin = codecs.open(rootpath+first_file+'/'+second_file+'/'+file, 'r',encoding ='utf-8')
							Text = fin.read()
						DOCS = re.findall(r'<DOC>.*?</DOC>', Text, re.S)
						for DOC in DOCS:
							docid = re.findall(r'<DOCNO>(.*?)</DOCNO>',DOC,re.S)
							if docid !=[]:
								docid = docid[0].strip()
								if docid in docnos:
									doc_list = []
									DOC = DOC.replace('\t',' ').replace('\n',' ').replace('\r', ' ').lower()
									Text = re.findall(r'<text>(.*?)</text>',DOC)
									head = re.findall(r'<head>(.*?)</head>',DOC)
									title = re.findall(r'<title>(.*?)</title>',DOC)
									hl = re.findall(r'<hl>(.*?)</hl>',DOC)
									Title = ' '.join(head)+' '+' '.join(title)+' '+' '.join(hl)
									Title = re.sub(r'<[^>]*?>',' ',Title)
									doc_list.append(nltk.word_tokenize(clean(Title)))
									if Text != []:
										text = re.sub(r'<[^>]*?>',' ',Text[0])
										sents = nltk.sent_tokenize(text)
										for sent in sents:
											words = nltk.word_tokenize(clean(sent))
											doc_list.append(words)
									else:
										Miss.append(docid)
									if doc_list != []:
										saveObjToFile(Locdir+'/big2datas/docs_disk12/'+docid,doc_list)
									else:
										Miss.append(docid)
	saveObjToFile('disk12.miss',Miss)
	print len(Miss)#'''

#对文本做词干化
if __name__ == "__main__":
	Locdir = os.getcwd()#/home/qiyy/Evaluation_Metrics
	krovetzstemmer = Stemmer()
	Stopwords = list(set(stopwords.words('english'))-set(['ma','most','against']))
	# blogs06()
	# WT10G()
	# blogs06_misstags()
	# AP8889_stem()
	# AP8889_kick()
