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

def doc2vec_feat(input_text):
	train_corpus = list(read_corpus(input_text))
	model = gensim.models.doc2vec.Doc2Vec()
	print ("doc2vec initiliased")
	model.build_vocab(train_corpus)
	print ("vocab built")
	model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
	print ("training done")
	corpusVectors = []
	for doc_id in range(len(train_corpus)):
	    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
	    corpusVectors.append(inferred_vector)
	return corpusVectors
