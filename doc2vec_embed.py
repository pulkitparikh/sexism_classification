import numpy as np
import gensim
import os
import collections
import smart_open
import random
import csv
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def read_corpus(input_data, tokens_only=False):
	for i in range(len(input_data)):
		yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(input_data[i]), [i])
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    #     cnt = 0
    #     for i, line in enumerate(f):
    #         if tokens_only:
    #             yield gensim.utils.simple_preprocess(line)
    #         else:
    #             # print gensim.utils.simple_preprocess(line)
    #             # raw_input()
    #             # For training data, add tags
    #             yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
    #         cnt += 1
    #         if cnt == 1:
    #         	break   

# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
# lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
# lee_test_file = test_data_dir + os.sep + 'lee.cor'

# print lee_train_file
def doc2vec_feat (input_text):
	train_corpus = list(read_corpus(input_text))
	model = gensim.models.doc2vec.Doc2Vec()
	print ("doc2vec initiliased")
	model.build_vocab(train_corpus)
	print ("vocab built")
	model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
	print ("training done")
	# inferred_vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
	corpusVectors = []
	for doc_id in range(len(train_corpus)):
	    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
	    # print len(inferred_vector)
	    corpusVectors.append(inferred_vector)
	return corpusVectors
	#print (corpusVectors)