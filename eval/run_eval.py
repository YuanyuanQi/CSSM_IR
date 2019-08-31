import os
import sys
import json
import uuid
import datetime
import subprocess

#command line : python3 run_eval.py ../bioasq_data/bioasq.test.json ../models/drmm/posit_test_preds.json | egrep '^map |^P_20 |^ndcg_cut_20 '
def format_bioasq2treceval_qrels(bioasq_data, filename):
	with open(filename, 'w') as f:
		for q in bioasq_data['questions']:
			for d in q['documents']:
				f.write('{0} 0 {1} 1'.format(q['id'], d))
				f.write('\n')

def format_bioasq2treceval_qret(bioasq_data, system_name, filename):
	with open(filename, 'w') as f:
		for q in bioasq_data['questions']:
			rank = 1
			for d in q['documents']:
				sim = (len(q['documents']) + 1 - rank) / float(len(q['documents']))
				f.write('{0} {1} {2} {3} {4} {5}'.format(q['id'], 0, d, rank, sim, system_name))
				f.write('\n')
				rank += 1

def trec_evaluate(qrels_file, qret_file):
	trec_eval_res = subprocess.Popen(
		[os.path.dirname(os.path.realpath(__file__)) + '/./trec_eval', '-m', 'all_trec', qrels_file, qret_file],
		stdout=subprocess.PIPE, shell=False)

	(out, err) = trec_eval_res.communicate()
	trec_eval_res = out.decode("utf-8")
	print(trec_eval_res)

if __name__ == '__main__':
#python3 run_eval.py ../bioasq_data/bioasq.test.json ../models/drmm/bio/posit_test_preds.json | egrep '^map |^P_20 |^ndcg_cut_20 '
#python3 run_eval.py /home/qiyy/Deep-relevance-ranking-master/robust04_data/split_5/rob04.test.s5.json /home/qiyy/Deep-relevance-ranking-master/models/drmm/posit_test_preds.json
	try:
		golden_file = sys.argv[1]#'/home/qiyy/Deep-relevance-ranking-master/robust04_data/split_5/rob04.test.s5.json'#
		predictions_file = sys.argv[2]#'/home/qiyy/Deep-relevance-ranking-master/models/drmm/posit_test_preds.json'
	except:
		sys.exit("Provide golden and predictions files.")
	
	try:
		system_name = sys.argv[3]
	except :
		try:
			system_name = predictions_file.split('/')[-1]
		except:
			system_name = predictions_file

	with open(golden_file, 'r') as f:
		golden_data = json.load(f)

	with open(predictions_file, 'r') as f:
		predictions_data = json.load(f)

	temp_dir = uuid.uuid4().hex
	qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
	qret_temp_file = '{0}/{1}'.format(temp_dir, 'qret.txt')

	try:
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		else:
			sys.exit("Possible uuid collision")

		format_bioasq2treceval_qrels(golden_data, qrels_temp_file)
		format_bioasq2treceval_qret(predictions_data, system_name, qret_temp_file)

		trec_evaluate(qrels_temp_file, qret_temp_file)
	finally:
		pass
		os.remove(qrels_temp_file)
		os.remove(qret_temp_file)
		os.rmdir(temp_dir)

