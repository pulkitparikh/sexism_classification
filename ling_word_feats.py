import pandas as pd
import csv
import os
import numpy as np
def loaddata_feat(ling_path,file_name,sind,nind,sepa):
	filepath1 =  ling_path + file_name
	embed_dict = {}
	liv = []
	if filepath1.endswith("NRC-VAD-Lexicon.txt") == False:
		emolex_df = pd.read_csv(filepath1,  names=["word", "emotion", "association"], sep=sepa)
		emolex_words = emolex_df.pivot_table(index='word', columns='emotion', values='association', aggfunc=np.mean).reset_index()
		liv  = np.round(emolex_words.mean(), decimals=2)
		liv = [str(i) for i in liv]
		base_filename = ling_path + 'feat_words.txt'
		with open(os.path.join(base_filename),'w') as outfile:
			emolex_words.to_string(outfile)
		if filepath1 == ling_path + "word_feat.txt":
			liv = 10 *['0.0']
		filepath1 = base_filename
	else:
		data = pd.read_csv(filepath1, sep='\t') 
		liv  = np.round(data.mean(), decimals=2)
		liv = [str(i) for i in liv]
	file = open(filepath1,'r')
	for line in file.readlines():
		row = line.strip().split()
		embed_dict[row[sind]] = row[nind:]

	file.close()
	return embed_dict, liv

def word_dict(emo_dict,word_embed_dict,word_neut,data):
	if data in word_embed_dict:
		for position, name in enumerate(word_embed_dict[data]):
			if name == 'NaN':
				word_embed_dict[data][position] = word_neut[position]
		emo_dict_new = emo_dict + word_embed_dict[data]
	else:
		emo_dict_new = emo_dict + word_neut
	return emo_dict_new

def perma_dict(emo_dict,perma_embed_dict,perma_neut, word_embed_dict, word_neut,data):
	if data in perma_embed_dict:
		for position, name in enumerate(perma_embed_dict[data]):
			if name == 'NaN':
				perma_embed_dict[data][position] = perma_neut[position]				
		emo_dict = emo_dict + perma_embed_dict[data]
		p_dict = word_dict(emo_dict,word_embed_dict,word_neut,data)
	else:
		emo_dict = emo_dict + perma_neut
		p_dict = word_dict(emo_dict,word_embed_dict,word_neut,data)
	return p_dict

def make_dict(emo_dict,sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut,data):
	if data in sentiment_embed_dict:
		emo_dict = emo_dict + sentiment_embed_dict[data]
		m_dict = perma_dict(emo_dict,perma_embed_dict,perma_neut, word_embed_dict, word_neut,data)
	else:
		emo_dict = emo_dict + sen_neut
		m_dict = perma_dict(emo_dict,perma_embed_dict,perma_neut, word_embed_dict, word_neut,data)
	return m_dict

def load_ling_word_vec_dicts(data_fold_path):
	ling_path = data_fold_path + "word_sent_embed/ling_resources/"
	emotion_embed_dict, neut = loaddata_feat(ling_path, 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',1,2,'\t')
	sentiment_embed_dict, sen_neut = loaddata_feat(ling_path, 'NRC-VAD-Lexicon.txt',0,1,'\t')
	perma_embed_dict, perma_neut = loaddata_feat(ling_path, 'permaV3_dd.csv',1,2,',')
	word_embed_dict, word_neut = loaddata_feat(ling_path, 'word_feat.txt',1,2,'\t')
	ling_word_vec_dim = 33
	return emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim
    
def emb_dict(emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut,data):
	text = []
	if data in emotion_embed_dict:
		text = emotion_embed_dict[data]
		final_list = make_dict(text,sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut,data)			
	else:
		text = neut
		final_list = make_dict(text,sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut,data)
	return final_list         

def ling_word_feat_sen_posts(data_sen, max_num_sent, max_word_count_per_sent, emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim, dir_filepath):
	for ID, post in enumerate(data_sen):
		feats_ID = np.zeros((max_num_sent, max_word_count_per_sent, ling_word_vec_dim))
		l = min(max_num_sent,len(post))
		for sen_ind, sen in enumerate(post[:l]):
			words = sen.split(' ')
			l_w = min(max_word_count_per_sent,len(words))
			for w_ind, w in enumerate(words[:l_w]):
				feats_ID[sen_ind, w_ind, :] = emb_dict(emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, w)
		np.save(dir_filepath + str(ID) + '.npy', feats_ID)			

def ling_word_feat_posts(data, max_post_length, emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim, dir_filepath):
	for ID, post in enumerate(data):
		feats_ID = np.zeros((max_post_length, ling_word_vec_dim))
		words = post.split(' ')
		l = min(max_post_length,len(words))
		for w_ind, w in enumerate(words[:l]):
			feats_ID[w_ind, :] = emb_dict(emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, w)
		np.save(dir_filepath + str(ID) + '.npy', feats_ID)			
