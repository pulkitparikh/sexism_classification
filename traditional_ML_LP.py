import numpy as np
import os, sys, pickle, csv, sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from eval_measures import *
from load_preproc import *
from string import punctuation
from word_embed import *
import nltk
from collections import Counter
from doc2vec_embed import *
from nltk.tokenize import TweetTokenizer
import time

def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict

def get_embeddings_dict(vector_type, emb_dim):
    if vector_type == 'sswe':
        emb_dim==50
        sep = '\t'
        vector_file = '../sswe-h.txt'
    elif vector_type =="glove":
        sep = ' '
        vector_file = '../../glove.6B/glove.6B.' + str(emb_dim) + 'd.txt'
    
    embed = get_embedding_weights(vector_file, sep)
    
    return embed

def classification_model(X_train, X_test, y_train, y_tested, model_type,bac_map):
    print ("Model Type:", model_type)

    model = get_model(model_type)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_predict = powerset_vec_to_label_lists(y_pred,bac_map)

    EM = exact_match(y_predict,y_tested)
    JC = jaccard_index_avg(y_predict,y_tested)
    IHL = inverse_hamming_loss(y_predict,y_tested)
    FI = F_metrics_instance(y_predict,y_tested)
    Fmicro = F_metrics_label_micro(y_predict,y_tested)
    Fmacro = F_metrics_label_macro(y_predict,y_tested)
    
    print ("exact match", EM)
    print ("jaccard", JC)
    print ("IHL", IHL)
    print ("F_mtrics_instance",   FI[2])
    print ("F_metrics_label_micro", Fmicro[2])
    print ("F_metrics_label_macro", Fmacro[2])

def get_model(m_type):
    if m_type == 'logistic regression':
        logreg = LogisticRegression()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=conf_dict_com['n_estimators'], n_jobs=-1, class_weight=conf_dict_com['class_weight'])
    elif m_type == "svm":
        logreg = LinearSVC(C=conf_dict_com['c_linear_SVC'],class_weight = conf_dict_com['class_weight'])
    elif m_type == "GBT":
        logreg = GradientBoostingClassifier(n_estimators= conf_dict_com['n_estimators'])
    else:
        print ("ERROR: Please specify a correst model")
        return None
    return logreg

def tf_idf(input_train,input_test,count_vec):
    tfidf_transformer = TfidfTransformer(norm = 'l2')
    bow_transformer_train= count_vec.fit_transform(input_train)
    bow_transformer_test =count_vec.transform(input_test)
    train_features = tfidf_transformer.fit_transform(bow_transformer_train).toarray()
    test_features= tfidf_transformer.transform(bow_transformer_test).toarray()
    return train_features,test_features

def feat_conact(features_word,features_char,features_POS,doc_feat,len_post,adj,text):
    features = []
    for i in range(len(text)): 
            features_text = np.append(features_word[i], features_char[i])
            features_text = np.append(features_text, features_POS[i])
            features_text = np.append(features_text, doc_feat[i])
            features_text = np.append(features_text, [len_post[i], adj[i]])
            features.append(features_text)
    return features

def get_features(Tdata,emb,emb_size):
    features = []
    tknzr = TweetTokenizer()    
    for i in range(len(Tdata)):
            concat = np.zeros(emb_size)
            Tdata[i] = Tdata[i].lower()
            text = ''.join([c for c in Tdata[i] if c not in punctuation])               
            tok = tknzr.tokenize(text)
            toklen = 1
            for wor in range(len(tok)):
                if tok[wor] in emb:
                        toklen += 1
                        flist = [float(i) for i in emb[str(tok[wor])]]
                        concat= flist + concat
            concat = concat/toklen
            features.append(concat)
    return features

def train(data_dict, labels, MODEL_TYPE,feat_type, bac_map,n_class,conf_dict_com):

    print (conf_dict_com['feat_type'])
    if conf_dict_com['feat_type']== "wordngrams":
        print("Using word based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="word",max_features = conf_dict_com['MAX_FEATURES'],stop_words='english',ngram_range = (1,2))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )
    elif conf_dict_com['feat_type'] == "charngrams": 
        print("Using char n-grams based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="char",max_features = conf_dict_com['MAX_FEATURES'], ngram_range = (1,5))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )         
    elif conf_dict_com['feat_type'] =="glove":
        print("Using glove embeddings")
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['glove']
        emb = get_embeddings_dict(conf_dict_com['feat_type'], emb_size)
        train_features = get_features(data_dict['text'][0:data_dict['train_en_ind']],emb,emb_size)
        test_features = get_features(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],emb,emb_size)
    elif conf_dict_com['feat_type'] == "elmo":
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['elmo']
        print("using elmo")
        train_features=[]
        test_features =[]
        for i in range(len(data_dict['text'][0:data_dict['train_en_ind']])):
            arr = np.load(conf_dict_com['filepath']+ str(i) + '.npy')
            avg_words = np.mean(arr, axis=0)
            train_features.append(avg_words)
        train_features = np.asarray(train_features)
        print (train_features.shape)
        inc = data_dict['test_st_ind']
        for i in range(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])):
            arr = np.load(conf_dict_com['filepath'] + str(inc) + '.npy')
            inc = inc + 1
            avg_words = np.mean(arr, axis=0)
            test_features.append(avg_words)
        test_features = np.asarray(test_features)
        print (test_features.shape)       
    elif conf_dict_com['feat_type'] == "ling_feat":
        len_post_train = []
        len_post_test = []
        POS_train = []
        POS_test = []
        adj_train = []
        adj_test =[]
        # doc2vec  features
        doc_feat_train = doc2vec_feat(data_dict['text'][0:data_dict['train_en_ind']])
        doc_feat_test = doc2vec_feat(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        # word ngrams and char ngrams
        count_vec_word = CountVectorizer(analyzer="word",max_features = conf_dict_com['MAX_FEATURES'],stop_words='english',ngram_range = (1,3))
        train_features_word, test_features_word = tf_idf(data_dict['text'][0:data_dict['train_en_ind']],data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],count_vec_word)
        count_vec_char = CountVectorizer(analyzer="char",max_features = conf_dict_com['MAX_FEATURES'], ngram_range = (3,5))
        train_features_char, test_features_char = tf_idf(data_dict['text'][0:data_dict['train_en_ind']],data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],count_vec_char)
        # linguistic features and POS
        for i in range(len(data_dict['text'][0:data_dict['train_en_ind']])):
            rep_te = data_dict['text'][i].replace(" ","")
            len_post_train.append(len(rep_te))
            adjectives_train =[token for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][i])) if pos.startswith('JJ')]
            counts_train = Counter(adjectives_train)
            adj_train.append(len(dict(counts_train.items())))
            POS_train.append([token + "_" + pos for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][i]))])
        ind = data_dict['test_st_ind']
        for j in range(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])):
            rep_tex = data_dict['text'][ind].replace(" ","")
            len_post_test.append(len(rep_tex))
            adjectives_test =[token for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][ind])) if pos.startswith('JJ')]
            counts_test = Counter(adjectives_test)
            adj_test.append(len(dict(counts_test.items())))
            POS_test.append([token + "_" + pos for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][ind]))])
            ind = ind + 1
        POS_traindata = [' '.join(i) for i in POS_train]
        POS_testdata = [' '.join(i) for i in POS_test]
        train_features_POS, test_features_POS = tf_idf(POS_traindata,POS_testdata,count_vec_word)
        train_features = feat_conact(train_features_word,train_features_char,train_features_POS,doc_feat_train,len_post_train,adj_train,data_dict['text'][0:data_dict['train_en_ind']])
        test_features = feat_conact(test_features_word,test_features_char,test_features_POS,doc_feat_test,len_post_test,adj_test,data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        print (np.shape(train_features))
        print (np.shape(test_features))
    if(MODEL_TYPE != "all"):
        print (MODEL_TYPE)
        classification_model(train_features, test_features, labels, data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], MODEL_TYPE, bac_map)
    else:
        for model_name in conf_dict_com['models']:
            print (model_name)
            classification_model(train_features, test_features, labels, data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], model_name, bac_map)


start = time.time()
conf_dict_list, conf_dict_com = load_config(sys.argv[1])

data_dict = load_data(conf_dict_com['filename'], conf_dict_com['data_path'], conf_dict_com['save_path'], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com['test_mode'])
labels,n_class,bac_map,for_map =fit_trans_labels_powerset(data_dict['lab'][:data_dict['train_en_ind']])

train(data_dict, labels, conf_dict_com['MODEL_TYPE'],conf_dict_com['feat_type'],bac_map,n_class,conf_dict_com)
timeLapsed = int(time.time() - startTime + 0.5)
t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
print(t_str)