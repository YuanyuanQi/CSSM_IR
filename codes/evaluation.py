#!/usr/bin/python
# -*- coding: UTF-8 -*-



import uuid
import subprocess
import os


def format_bioasq2treceval_qrels(qrel_data, filename):
	with open(filename, 'w') as f:
		# for q in bioasq_data['questions']:
			# for d in q['documents']:
				# f.write('{0} 0 {1} 1'.format(q['id'], d))
				# f.write('\n')
		for qid,value in qrel_data.items():
			for docno in value['docno']:
				f.write('{0} 0 {1} 1'.format(qid, docno))
				f.write('\n')

def format_bioasq2treceval_qret(qret_data, filename):
	with open(filename, 'w') as f:
		for q in qret_data['questions']:
			rank = 0
			# print q
			for d in q['documents']:
				# sim = (len(q['documents']) + 1 - rank) / float(len(q['documents']))
				sim =  q['score'][rank]
				f.write('{0} {1} {2} {3} {4} {5}'.format(q['id'], 0, d, rank, sim,'window_top2-bm25' ))
				f.write('\n')
				rank += 1

def trec_evaluate(qrels_file, qret_file, eval_name):
	trec_eval_res = subprocess.Popen(#os.path.dirname(os.path.realpath(__file__)) + '/./trec_eval
		['../code/eval/trec_eval', '-m', 'all_trec', qrels_file, qret_file],
		stdout=subprocess.PIPE, shell=False)
	(out, err) = trec_eval_res.communicate()
	trec_eval_res = out.decode("utf-8")
	print '\n'.join(trec_eval_res.split('\n')[5:6])
	file = open(eval_name, 'w')
	file.write(trec_eval_res)
	file.close()

def trec(golden_data, predictions_data, evalfile_name):
	temp_dir = uuid.uuid4().hex
	qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
	qret_temp_file = '{0}/{1}'.format(temp_dir, 'qret.txt')
	try:
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		else:
			sys.exit("Possible uuid collision")

		format_bioasq2treceval_qrels(golden_data, qrels_temp_file)
		format_bioasq2treceval_qret(predictions_data, qret_temp_file)# qret_temp_file = '../Robust04.BM25b0.35_1.res'
		print evalfile_name
		trec_evaluate(qrels_temp_file, qret_temp_file, evalfile_name)

	finally:
		pass
		os.remove(qrels_temp_file)
		os.remove(qret_temp_file)
		os.rmdir(temp_dir)