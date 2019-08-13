import torch
import os
import sys
import numpy as np
import h5py
sys.path.insert(0, '/home/pulkit/research/InferSent')
from models import InferSent
import tensorflow as tf
import tensorflow_hub as hub
from bert_serving.client import ConcurrentBertClient
from keras.models import load_model

def tuned_embed_posts(posts, max_sent_cnt, org_embed_dim, embed_dim, mod_name, base_feat_Name, base_feat_filename, func_name, use_saved_sent_feats, data_fold_path, save_fold_path):
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	if use_saved_sent_feats and os.path.isfile(base_feat_filename):
		print("loading %s sent feats" % base_feat_Name)
		with h5py.File(base_feat_filename, "r") as hf:
			feats = hf['feats'][:]
	else:
		print("computing %s sent feats" % base_feat_Name)
		feats = func_name(posts, max_sent_cnt, org_embed_dim, data_fold_path)

	encoder = load_model(save_fold_path + 'mod_aut_' + mod_name + '.h5')	
	for ind, sentences in enumerate(posts):
		l = min(max_sent_cnt,len(sentences))
		embeddings = feats[ind, :l, :]
		posts_arr[ind, :l, :] = encoder.predict(embeddings)
	return posts_arr

# run the command below on a seperate terminal before running the function
# screen -L bert-serving-start -model_dir ../bert/uncased_L-24_H-1024_A-16/  -num_worker=1 -max_seq_len=None -device_map=0
# os.system('')
# screen -L bert-serving-start -model_dir=../bert/uncased_L-12_H-768_A-12 -tuned_model_dir=../bert/tmp/pretraining_output/ -ckpt_name=model.ckpt-100000 -num_worker=1 -max_seq_len=None -device_map=0

def bert_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	bc = ConcurrentBertClient()
	for ind, sentences in enumerate(posts):
		embeddings = bc.encode(sentences)
		l = min(max_sent_cnt,len(sentences))
		posts_arr[ind, :l, :] = embeddings[:l]
		if ind % 1000 == 0:
			print("batch %s of %s done" % (ind, len(posts)))
	return posts_arr

def use_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	with tf.Graph().as_default():
		embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
		messages = tf.placeholder(dtype=tf.string)
		output = embed(messages)

		with tf.Session(config=config) as session:
			session.run([tf.global_variables_initializer(), tf.tables_initializer()])
			for ind, sentences in enumerate(posts):
				embeddings = session.run(output, feed_dict={messages: sentences})
				l = min(max_sent_cnt,len(sentences))
				posts_arr[ind, :l, :] = embeddings[:l]
	return posts_arr

def infersent_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	model_path = data_fold_path + 'word_sent_embed/infersent2.pickle'
	word_emb_path = data_fold_path + 'word_sent_embed/fasttext.vec'
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))

	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
	                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(model_path))
	model.set_w2v_path(word_emb_path)

	all_sents = []
	for sens in posts:
		all_sents.extend(sens)
	
	model.build_vocab(all_sents, tokenize=False)

	for ind, sentences in enumerate(posts):
		embeddings = model.encode(sentences, tokenize=False, verbose=False)
		l = min(max_sent_cnt,len(sentences))
		posts_arr[ind, :l, :] = embeddings[:l]

	return posts_arr

# p = [["everyday sexism is rampant", "we should not do that"], ["we should not do that"], ["everyday is rampant", "we should",  "not do that"], ["a", "b", "c", "d"]]
# o = bert_embed_posts(p, 3, 1024, 35)
# print(o)

# o1 = inferSent_embed_posts_raw(p, 'word_vectors/glove.6B.300d.txt', 'data/infersent1.pkl', 2)

# from loadPreProc import *
# import pickle
# conf_dict = load_config("data/config1.txt")
# f_str =  conf_dict["filename"][:-4].split("/")[-1]
# cl_in_filename = ("%scl_input_%s_%s_%s_%s.pickle" % (conf_dict["data_folder_name"], f_str, conf_dict['TEST_RATIO'], conf_dict['VALID_RATIO'], conf_dict['RANDOM_STATE']))
# if conf_dict["use_saved_cleaned_input"] and os.path.isfile(cl_in_filename):
#     print "loading cleaned input"
#     with open(cl_in_filename, 'rb') as f_cl_in:
#         data_dict, train_clean, val_clean, test_clean, train_sen_clean, val_sen_clean, test_sen_clean, trainX, valX, testX, trainX_sen, valX_sen, testX_sen, vocab = pickle.load(f_cl_in)
# # o1 = inferSent_embed_posts_raw(train_sen_clean[:100], 'word_vectors/glove.6B.300d.txt', 'data/infersent1.pkl', trainX_sen.shape[1])
# o1 = use_embed_posts(train_sen_clean[:100], trainX_sen.shape[1])
# print o1
def sent_featurize(sent_feats_raw, data_dict, poss_sent_feats_emb_dict, use_saved_sent_feats, save_sent_feats, data_fold_path, save_fold_path, test_mode):
	max_num_sent_feats = 3
	max_num_attributes = 2
	sent_feats = []
	sent_feat_str = ''
	for sent_feat_raw_dict in sent_feats_raw:
		feat_name = sent_feat_raw_dict['emb']
		sent_feat_str += ("%s~%s~" % (sent_feat_raw_dict['emb'], sent_feat_raw_dict['m_id']))

		sent_feat_dict ={}
		for sent_feat_attr_name, sent_feat_attr_val in sent_feat_raw_dict.items():
			sent_feat_dict[sent_feat_attr_name] = sent_feat_attr_val

		print("computing %s sent feats; test_mode = %s" % (feat_name, test_mode))

		s_filename = ("%ssent_feat~%s.h5" % (save_fold_path, feat_name))
		if use_saved_sent_feats and os.path.isfile(s_filename):
			print("loading %s sent feats" % feat_name)
			with h5py.File(s_filename, "r") as hf:
				sent_feat_dict['feats'] = hf['feats'][:data_dict['test_en_ind']]
		else:
			if feat_name.startswith('bert'):
				if feat_name == 'bert' or feat_name.startswith('bert_pre'):
					feats = bert_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_feats_emb_dict[feat_name], data_fold_path)
				else:
					feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_feats_emb_dict['bert'], poss_sent_feats_emb_dict[feat_name], feat_name, 'bert', ("%ssent_feat~bert.h5" % (save_fold_path)), bert_embed_posts, use_saved_sent_feats, data_fold_path, save_fold_path)
			elif feat_name.startswith('use'):
				if feat_name == 'use':
					feats = use_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_feats_emb_dict[feat_name], data_fold_path)
				else:
					feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_feats_emb_dict['use'], poss_sent_feats_emb_dict[feat_name], feat_name, 'use', ("%ssent_feat~use.h5" % (save_fold_path)), use_embed_posts, use_saved_sent_feats, data_fold_path, save_fold_path)
			elif feat_name.startswith('infersent'):
				if feat_name == 'infersent':
					feats = infersent_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_feats_emb_dict[feat_name], data_fold_path)
				else:
					feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_feats_emb_dict['infersent'], poss_sent_feats_emb_dict[feat_name], feat_name, 'infersent', ("%ssent_feat~infersent.h5" % (save_fold_path)), infersent_embed_posts, use_saved_sent_feats, data_fold_path, save_fold_path)

			sent_feat_dict['feats'] = feats[:data_dict['test_en_ind']]

			if save_sent_feats:
				print("saving %s sent feats" % feat_name)
				with h5py.File(s_filename, "w") as hf:
					hf.create_dataset('feats', data=feats)

		sent_feats.append(sent_feat_dict)

	sent_feat_str += "~" * ((max_num_sent_feats - len(sent_feats_raw)) * max_num_attributes)

	return sent_feats, sent_feat_str[:-1]