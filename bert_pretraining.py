import os
import re
from nltk import sent_tokenize, word_tokenize
import pickle
import csv

def remove_test_from_unlab(filename, op_name, fin_data_name, all_data_file, data_path, save_path):
	raw_fname = data_path + filename
	all_data_fname = data_path + all_data_file
	s_name = data_path + op_name
	data_fname = save_path + fin_data_name
	r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
	r_white = re.compile(r'[\s.(?)!]+')

	with open(data_fname, 'rb') as f_data:
		data_dict = pickle.load(f_data)
	# removing validation and test data
	test_data = data_dict['text'][data_dict['test_st_ind']:]

	all_id_dict = {}
	with open(all_data_fname, 'r') as allfile:
		reader = csv.DictReader(allfile, delimiter = '\t')
		for row in reader:
			a_clean = r_white.sub(' ', r_anum.sub('', row['post'].lower())).strip()
			all_id_dict[a_clean] = row['post id']
	print('all_id_dict done')

	test_id_dict= {}
	for t_post in test_data:
		test_id_dict[all_id_dict[t_post]] = True
	print('test_id_dict done')

	num_remove = 0
	num_remain = 0
	with open(raw_fname, 'r') as txtfile:
		with open(s_name, 'w') as wfile:
			reader = csv.DictReader(txtfile, delimiter = '\t')
			for row in reader:
				row_clean = r_white.sub(' ', r_anum.sub('', row['post'].lower())).strip()
				test_flag = False
				for test_post in test_data:
					if test_post in row_clean:
						test_flag = True
						break
				if test_flag or row['id'][5:] in test_id_dict:
					num_remove += 1
				else:
					wfile.write("%s\n" % row['post'])
					num_remain += 1
	print("removed: %s, remaining %s" % (num_remove, num_remain))
	print("test data size: %s" % (len(test_data)))

# remove_test_from_unlab('unlab_data_postids.csv', 'unlab_sans_test.txt', 'data_0.15_0.15_22_35_False.pickle', 'data.csv', 'data/', 'saved/')

def bert_pretraining_data(filename, data_path, save_path):
	max_seq_len = 1000
	raw_fname = data_path + filename
	s_name = save_path + filename[:-4] + '_bert_pre.txt'
	if os.path.isfile(s_name):
		print("already exists")
	else:
		with open(raw_fname, 'r') as txtfile:
			with open(s_name, 'w') as wfile:
				post_cnt = 0
				for post in txtfile.readlines():
					list_sens = []
					post_has_big_sens = False
					for se in sent_tokenize(post):
						if len(word_tokenize(se)) > max_seq_len:
							post_has_big_sens = True
							break
						list_sens.append(se)
					if post_has_big_sens:
						continue

					if post_cnt > 0:
						wfile.write("\n")
					for se in list_sens:
						wfile.write("%s\n" % se)

					post_cnt += 1
		print("saved %d bert pretraining data" % post_cnt)
# bert_pretraining_data('unlab_sans_test.txt', "data/", "../bert/tmp/")

# screen -L python create_pretraining_data.py \
#   --input_file=tmp/unlab_sans_test_bert_pre.txt \
#   --output_file=tmp/tf_examples.tfrecord \
#   --vocab_file=../bert/uncased_L-12_H-768_A-12/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=120 \
#   --max_predictions_per_seq=18 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

# screen -L python run_pretraining.py \
#   --input_file=tmp/tf_examples.tfrecord \
#   --output_dir=tmp/pretraining_output \
#   --do_train=True \
#   --do_eval=True \
#   --bert_config_file=../bert/uncased_L-12_H-768_A-12/bert_config.json \
#   --init_checkpoint=../bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
#   --train_batch_size=25 \
#   --max_seq_length=120 \
#   --max_predictions_per_seq=18 \
#   --num_train_steps=100000 \
#   --num_warmup_steps=10000 \
#   --learning_rate=2e-5