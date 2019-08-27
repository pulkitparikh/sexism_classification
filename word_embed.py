import os
import numpy as np
import pickle
import h5py
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from allennlp.commands.elmo import ElmoEmbedder
from ling_word_feats import load_ling_word_vec_dicts, ling_word_feat_sen_posts, ling_word_feat_posts
from load_preproc import is_model_hier 

def init_comb_vocab_word_list(data_dict):
	comb_vocab_word_list_dict = {}
	for post in data_dict['text']:
		words = post.split(' ')
		for w in words:
			comb_vocab_word_list_dict[w] = None
	return comb_vocab_word_list_dict

def comb_vocab_dict(feat_name, embed_size, data_dict, test_mode, data_fold_path, save_fold_path, use_saved_word_feats, save_word_feats):
	voc_filename = ("%scomb_vocab~%s.pickle" % (save_fold_path, feat_name))
	if use_saved_word_feats and os.path.isfile(voc_filename):
		print("loading comb vocab for %s" % feat_name)
		with open(voc_filename, "rb") as f:
			comb_vocab = pickle.load(f)
	else:
		filename = ("%scomb_word_list.pickle" % save_fold_path)
		if os.path.isfile(filename):
			print("loading comb word list")
			with open(filename, "rb") as ff:
				comb_vocab_word_list_dict = pickle.load(ff)
		else:
			comb_vocab_word_list_dict = init_comb_vocab_word_list(data_dict)
			print("saving comb word list")
			with open(filename, "wb") as ff:
				pickle.dump(comb_vocab_word_list_dict, ff)

		comb_vocab = {}
		for w in comb_vocab_word_list_dict.keys():
			comb_vocab[w] = np.zeros((embed_size))

		if feat_name == 'glove':
		    vector_file = data_fold_path + 'word_sent_embed/glove.txt'
		elif feat_name == 'fasttext':
			vector_file = data_fold_path + 'word_sent_embed/fasttext.vec'
		    
		with open(vector_file, 'r') as f_e:
		    for line in f_e:
		        tokens = line.strip().split(' ')
		        if tokens[0] in comb_vocab:
		            comb_vocab[tokens[0]] = np.array(tokens[1:], dtype = 'float64')

		if save_word_feats:
			print("saving comb vocab for %s" % feat_name)
			with open(voc_filename, "wb") as f:
				pickle.dump(comb_vocab, f)
	return comb_vocab

def get_embed_mat(feat_name, embed_size, vocab, comb_vocab, test_mode, save_fold_path, use_saved_word_feats, save_word_feats):
	filename = ("%sembed_mat~%s~%s.pickle" % (save_fold_path, feat_name, test_mode))
	if use_saved_word_feats and os.path.isfile(filename):
		print("loading embed mat for %s; test mode = %s" % (feat_name, test_mode))
		with open(filename, "rb") as f:
			embed_mat = pickle.load(f)
	else:
		embed_mat = np.zeros((len(vocab), embed_size))
		vocab_core = dict(vocab)
		del vocab_core['UNK']
		del vocab_core['PAD']
		for word, w_ind in vocab_core.items():
			embed_mat[w_ind] = comb_vocab[word]
		if save_word_feats:
			print("saving embed mat for %s; test mode = %s" % (feat_name, test_mode))
			with open(filename, "wb") as f:
				pickle.dump(embed_mat, f)
	return embed_mat

def create_we_vec(data, max_post_length, embed_size, comb_vocab, dir_filepath):
	for ID, post in enumerate(data):
		feats_ID = np.zeros((max_post_length, embed_size))
		words = post.split(' ')
		l = min(len(words), max_post_length)
		for ind_w, w in enumerate(words[:l]):
			feats_ID[ind_w, :] = comb_vocab[w]
		np.save(dir_filepath + str(ID) + '.npy', feats_ID)			

def create_sen_we_vec(data_sen, max_num_sent, max_word_count_per_sent, embed_size, comb_vocab, dir_filepath):
	for ID, sentences in enumerate(data_sen):
		feats_ID = np.zeros((max_num_sent, max_word_count_per_sent, embed_size))
		l = min(len(sentences), max_num_sent)
		for ind_sen, sen in enumerate(sentences[:l]):
			words = sen.split(' ')
			l = min(len(words), max_word_count_per_sent)
			for ind_w, w in enumerate(words[:l]):
				feats_ID[ind_sen, ind_w, :] = comb_vocab[w]
		np.save(dir_filepath + str(ID) + '.npy', feats_ID)			

def prep_X_no_trainable(data, comb_vocab, embed_size, max_post_length, dir_filepath):
    create_we_vec(data, max_post_length, embed_size, comb_vocab, dir_filepath)

def prep_X_sent_no_trainable(data_sen, comb_vocab, embed_size, max_num_sent, max_word_count_per_sent, dir_filepath):
    create_sen_we_vec(data_sen, max_num_sent, max_word_count_per_sent, embed_size, comb_vocab, dir_filepath)

def prep_Xdata(data, t, max_post_length):
    pad_data = pad_sequences(t.texts_to_sequences(data), maxlen=max_post_length, padding='post', truncating = 'post')
    return pad_data

def transform_text_indices(data_sen_clean, max_num_sent, max_word_count_per_sent, t):
    dataX = np.zeros((len(data_sen_clean), max_num_sent, max_word_count_per_sent), dtype=np.uint32)
    for ind, post_sen in enumerate(data_sen_clean):
        l = min(max_num_sent,len(post_sen))
        dataX[ind, 0:l, :] = pad_sequences(t.texts_to_sequences(post_sen[:l]), maxlen=max_word_count_per_sent, padding='post', truncating = 'post')
    return dataX

def prep_Xdata_sent(data_sen, t, max_num_sent, max_word_count_per_sent):
    pad_data_sen = transform_text_indices(data_sen, max_num_sent, max_word_count_per_sent, t)
    return pad_data_sen

def prep_Xdata_both(var_hier, data_dict, test_mode, save_fold_path, dir_filepath):
	tok_filename = ("%stok_vocab~%s.pickle" % (save_fold_path, test_mode))
	if os.path.isfile(tok_filename):
		print("loading tok vocab; test mode = %s" % test_mode)
		with open(tok_filename, "rb") as tf:
			t, vocab = pickle.load(tf)
	else:
		t, vocab = tokenize_vocab(data_dict['text'][:data_dict['train_en_ind']])
		print("saving tok vocab; test mode = %s" % test_mode)
		with open(tok_filename, "wb") as tf:
			pickle.dump([t, vocab], tf)

	filename = ("%spad_data~%s~%s.h5" % (save_fold_path, var_hier, test_mode))
	if os.path.isfile(filename):
		print("loading pad data for %s; test mode = %s" % (var_hier, test_mode))
		with open(filename, "rb") as f:
			pad_data = pickle.load(f) 
	else:
		if var_hier == True:
			pad_data = prep_Xdata_sent(data_dict['text_sen'][:data_dict['test_en_ind']], t, data_dict['max_num_sent'], data_dict['max_words_sent'])
		else:
			pad_data = prep_Xdata(data_dict['text'][:data_dict['test_en_ind']], t, data_dict['max_post_length'])
		print("saving pad data for %s; test mode = %s" % (var_hier, test_mode))
		with open(filename, "wb") as f:
			pickle.dump(pad_data, f) 
	for ID, feats_ID in enumerate(pad_data):
		np.save(dir_filepath + str(ID) + '.npy', feats_ID)			
	return vocab, pad_data.shape[1:]

def elmo_apply_sen(ID, elmo_word_feat, data_dict, filepath, emb_size):
	saved_arr =  np.load(filepath)
	feats_ID = np.zeros((data_dict['max_num_sent'], data_dict['max_words_sent'], emb_size))
	cur_sen_st = 0
	for ind_sen in range(elmo_word_feat['n_sents_lim'][ID]):
		next_sen_st = cur_sen_st+elmo_word_feat['n_words_sent'][ID][ind_sen]
		feats_ID[ind_sen, :elmo_word_feat['n_words_sent'][ID][ind_sen], :] = saved_arr[cur_sen_st:next_sen_st, :]
		cur_sen_st = next_sen_st
	return feats_ID

def elmo_apply(ID, elmo_word_feat, data_dict, filepath, emb_size):
	saved_arr =  np.load(filepath)
	feats_ID = np.zeros((data_dict['max_post_length'], emb_size))
	feats_ID[:elmo_word_feat['n_words_lim'][ID],:] = saved_arr[:elmo_word_feat['n_words_lim'][ID],:]
	return feats_ID

def elmo_comp_stats(elmo_word_feat, data_dict):
	elmo_word_feat['n_sents_lim'] = np.zeros(len(data_dict['text']), dtype=np.uint16)
	elmo_word_feat['n_words_lim'] = np.zeros(len(data_dict['text']), dtype=np.uint16)
	elmo_word_feat['n_words_sent'] = np.zeros((len(data_dict['text']), data_dict['max_num_sent']), dtype=np.uint16)
	for ID, sentences in enumerate(data_dict['text_sen']):
		elmo_word_feat['n_sents_lim'][ID] = min(len(sentences), data_dict['max_num_sent'])
		word_len_post = len(data_dict['text'][ID].split(' '))
		elmo_word_feat['n_words_lim'][ID] = min(word_len_post, data_dict['max_post_length'])
		for ind_sen, sent in enumerate(sentences[:elmo_word_feat['n_sents_lim'][ID]]):
			sent_words = sent.split(' ')
			elmo_word_feat['n_words_sent'][ID][ind_sen] = len(sent_words)

def elmo_save_no_pad(data_dict, elmo, dir_filepath, emb_size):
	for ID, sentences in enumerate(data_dict['text_sen']):
		word_len_post = len(data_dict['text'][ID].split(' '))
		feats_ID = np.zeros((word_len_post, emb_size))
		w_ind = 0
		for ind_sen, sent in enumerate(sentences):
			sent_words = sent.split(' ')
			vectors = elmo.embed_sentence(sent_words)
			for i in range(len(sent_words)):
				feats_ID[w_ind] = np.concatenate((vectors[0][i], vectors[1][i], vectors[2][i]), axis=None)
				w_ind += 1
		np.save(dir_filepath + str(ID) + '.npy', feats_ID)			

def tokenize_vocab(train_data):
	t = Tokenizer(oov_token='UNK')
	t.fit_on_texts(train_data)
	vocab = t.word_index.copy()
	vocab['PAD'] = 0
	return t, vocab

def word_featurize(word_feats_raw, model_type, data_dict, poss_word_feats_emb_dict, use_saved_word_feats, save_word_feats, data_fold_path, save_fold_path, test_mode):
	max_num_word_feats = 4
	max_num_attributes = 4
	word_feats = []
	var_model_hier = is_model_hier(model_type)
	word_feat_str = ''
	for word_feat_raw_dict in word_feats_raw:
		word_feat_name = word_feat_raw_dict['emb']
		word_feat_str += ("%s~%s~%s~%s~" % (word_feat_raw_dict['emb'], word_feat_raw_dict['s_enc'], word_feat_raw_dict['m_id'], (str(word_feat_raw_dict['tune']) if 'tune' in word_feat_raw_dict else '')))

		word_feat_dict ={}
		for word_feat_attr_name, word_feat_attr_val in word_feat_raw_dict.items():
			word_feat_dict[word_feat_attr_name] = word_feat_attr_val

		if 'tune' in word_feat_raw_dict and word_feat_raw_dict['tune'] == True:
			var_tune = True
			di_filename = ("%sad_word_feat~%s~%s~%s~%s.pickle" % (save_fold_path, word_feat_name, var_model_hier, var_tune, test_mode))
		else:
			var_tune = False
			di_filename = ("%sad_word_feat~%s~%s~%s.pickle" % (save_fold_path, word_feat_name, var_model_hier, var_tune))

		comb_str = ("%s~%s~%s" % (word_feat_name, var_model_hier, var_tune))
		print("computing %s word feat comb; test_mode = %s" % (comb_str, test_mode))
	
		if use_saved_word_feats and os.path.isfile(di_filename):
			print("loading ad_word_feats for %s" % comb_str)
			with open(di_filename, "rb") as f:
				ad_word_feats = pickle.load(f)
		else:
			ad_word_feats = {}
			if word_feat_name == 'glove' or word_feat_name == 'fasttext':
				comb_vocab = comb_vocab_dict(word_feat_name, poss_word_feats_emb_dict[word_feat_name], data_dict, test_mode, data_fold_path, save_fold_path, use_saved_word_feats, save_word_feats)
				if var_tune == True:
					ad_word_feats['filepath'] = save_fold_path + 'word_vecs~' + word_feat_name + '/' + str(var_model_hier) + '/' + str(var_tune) + '/' + str(test_mode) + '/'
					os.makedirs(ad_word_feats['filepath'], exist_ok=True)
					vocab, ad_word_feats['dim_shape'] = prep_Xdata_both(var_model_hier, data_dict, test_mode, save_fold_path, ad_word_feats['filepath'])
					ad_word_feats['embed_mat'] = get_embed_mat(word_feat_name, poss_word_feats_emb_dict[word_feat_name], vocab, comb_vocab, test_mode, save_fold_path, use_saved_word_feats, save_word_feats)
				else:
					ad_word_feats['filepath'] = save_fold_path + 'word_vecs~' + word_feat_name + '/' + str(var_model_hier) + '/' + str(var_tune) + '/'
					os.makedirs(ad_word_feats['filepath'], exist_ok=True)
					if var_model_hier == True:
						prep_X_sent_no_trainable(data_dict['text_sen'], comb_vocab, poss_word_feats_emb_dict[word_feat_name], data_dict['max_num_sent'], data_dict['max_words_sent'], ad_word_feats['filepath'])
						ad_word_feats['dim_shape'] = [data_dict['max_num_sent'], data_dict['max_words_sent'], poss_word_feats_emb_dict[word_feat_name]]
					else:
						prep_X_no_trainable(data_dict['text'], comb_vocab, poss_word_feats_emb_dict[word_feat_name], data_dict['max_post_length'], ad_word_feats['filepath'])
						ad_word_feats['dim_shape'] = [data_dict['max_post_length'], poss_word_feats_emb_dict[word_feat_name]]
			elif word_feat_name == 'ling':
				ad_word_feats['filepath'] = save_fold_path + 'word_vecs~' + word_feat_name + '/' + str(var_model_hier) + '/' + str(var_tune) + '/'
				os.makedirs(ad_word_feats['filepath'], exist_ok=True)
				emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim = load_ling_word_vec_dicts(data_fold_path)
				if var_model_hier == True:
					ad_word_feats['dim_shape'] = [data_dict['max_num_sent'], data_dict['max_words_sent'], poss_word_feats_emb_dict[word_feat_name]]
					ling_word_feat_sen_posts(data_dict['text_sen'], data_dict['max_num_sent'], data_dict['max_words_sent'], emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim, ad_word_feats['filepath'])
				else:
					ad_word_feats['dim_shape'] = [data_dict['max_post_length'], poss_word_feats_emb_dict[word_feat_name]]
					ling_word_feat_posts(data_dict['text'], data_dict['max_post_length'], emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim, ad_word_feats['filepath'])
			elif word_feat_name == 'elmo':
				ad_word_feats['filepath'] = save_fold_path + 'word_vecs~' + word_feat_name + '/' + str(var_tune) + '/'
				if (use_saved_word_feats == False) or (not os.path.isfile(ad_word_feats['filepath'] + '0.npy')):
					os.makedirs(ad_word_feats['filepath'], exist_ok=True)
					elmo = ElmoEmbedder()
					elmo_save_no_pad(data_dict, elmo, ad_word_feats['filepath'], poss_word_feats_emb_dict[word_feat_name])
				ad_word_feats['emb_size'] = poss_word_feats_emb_dict[word_feat_name]
				elmo_comp_stats(ad_word_feats, data_dict)
				if var_model_hier == True:
					ad_word_feats['func'] = elmo_apply_sen
					ad_word_feats['dim_shape'] = [data_dict['max_num_sent'], data_dict['max_words_sent'], poss_word_feats_emb_dict[word_feat_name]]
				else:
					ad_word_feats['func'] = elmo_apply
					ad_word_feats['dim_shape'] = [data_dict['max_post_length'], poss_word_feats_emb_dict[word_feat_name]]
			if save_word_feats:
				print("saving ad_word_feats for %s" % comb_str)
				with open(di_filename, "wb") as f:
					pickle.dump(ad_word_feats, f)
		for k,v in ad_word_feats.items():
			word_feat_dict[k] = v
		word_feats.append(word_feat_dict)

	word_feat_str += "~" * ((max_num_word_feats - len(word_feats_raw)) * max_num_attributes)
	return word_feats, word_feat_str[:-1]	    		
