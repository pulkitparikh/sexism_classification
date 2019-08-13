import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import preprocessor as p
from loadPreProc import *
from evalMeasures import *
import sys

conf_dict_list, conf_dict_com = load_config(sys.argv[1])

# models = [ 'cnn', 'lstm', 'blstm', 'blstm_attention']
# word_vectors = ["random", "glove" ,"sswe"]
prob_trans_type = "None"
model_type = "random"
vector_type = "None"
embed_size = "None"

def train_evaluate_model(trainY_list, true_vals, metr_dict):
# def train_evaluate_model(trainY_list, true_vals, f_res, f_err_inst, f_err_lab, test_clean, test_sen_clean, train_labels):
	# r_op = np.random.randint(low = 0, high = 2, size=(len(trainY_list), NUM_CLASSES))

	train_coverage = np.zeros(NUM_CLASSES)
	for lset in trainY_list:
		for l in lset:
			train_coverage[l] += 1.0
	train_coverage /= float(len(trainY_list))
	print(train_coverage)
	print(np.mean(train_coverage))

	r_op = np.empty((len(trainY_list), NUM_CLASSES), dtype=int)
	for i in range(len(trainY_list)):
		while(True):
			r_op[i, :] = 0
			for j in range(NUM_CLASSES):
				r_num = np.random.uniform()
				# if r_num < 1:
				# if r_num < 0.5:
				if r_num < train_coverage[j]:
					r_op[i,j] = 1			
			if sum(r_op[i, :]) > 0:
				break
		# max_val = 0
		# max_ind = 0
		# for j in range(NUM_CLASSES):
		# 	r_num = np.random.uniform()
		# 	if r_num > max_val:
		# 		max_val = r_num
		# 		max_ind = j	
		# 	if r_num < 1:#train_coverage[j]:
		# 		r_op[i,j] = 1			
		# if sum(r_op[i, :]) == 0:
		# 	r_op[i,max_ind] = 1
	
	pred_vals = di_op_to_label_lists(r_op)

	# print(r_op[0:3])
	# print(pred_vals[0:3])
	# print(true_vals[0:3])
	# input()
	
	# max_label_card = np.amax([len(x) for x in trainY_list])
	# l_card_freqs = [0.0] * max_label_card
	# cum_card_freqs = [0.0] * max_label_card
	# for l in trainY_list:
	# 	l_card_freqs[len(l)-1] += 1
	
	# prev = 0
	# for i in range(max_label_card):
	# 	l_card_freqs[i] = l_card_freqs[i]/len(trainY_list)
	# 	cum_card_freqs[i] = l_card_freqs[i] + prev
	# 	prev = cum_card_freqs[i]

	# print (max_label_card)
	# print ([l*100 for l in l_card_freqs])
	# # print cum_card_freqs

	# pred_vals = []
	# for v in true_vals:
	# 	r_num = np.random.uniform()	
	# 	for ind, val in enumerate(cum_card_freqs):
	# 		if r_num < val:	
	# 			break
	# 	label_set = np.random.randint(0, NUM_CLASSES, ind+1).tolist()		
	# 	pred_vals.append(label_set)

	################# See if the label cardinality dist in pred_vals is similar to the train counterpart #############
	# max_label_card = np.amax([len(x) for x in pred_vals])
	# l_card_freqs = [0.0] * max_label_card
	# cum_card_freqs = [0.0] * max_label_card
	# for l in pred_vals:
	# 	l_card_freqs[len(l)-1] += 1
	
	# prev = 0
	# for i in range(max_label_card):
	# 	l_card_freqs[i] = l_card_freqs[i]/len(pred_vals)
	# 	cum_card_freqs[i] = l_card_freqs[i] + prev
	# 	prev = cum_card_freqs[i]

	# print (max_label_card)
	# print ([l*100 for l in l_card_freqs])
	################# end #############

	return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict)
	# return calc_metrics_print(pred_vals, true_vals, f_res, f_err_inst, f_err_lab, test_clean, test_sen_clean, train_labels)

# def get_train_test(x_text, labels):
    
#     X_train, X_test, Y_train, Y_test = train_test_split(x_text, labels, random_state=RANDOM_STATE, test_size=TEST_RATIO)
       
#     return X_train, Y_train, X_test, Y_test

data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"])

# f_res = open(conf_dict_com["output_folder_name"] + '/' + conf_dict_com["res_filename"], 'a')
# data_dict = load_data("data/"+conf_dict_com["filename"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'])
# print("data loaded")

# print("prob_trans: %s, model: %s, vector: %s, embedding size: %s\n" % (prob_trans_type, model_type, vector_type, embed_size))
# f_res.write("prob_trans: %s, model: %s, vector: %s, embedding size: %s\n\n" % (prob_trans_type, model_type, vector_type, embed_size))

# f_err_inst = open("results/dummy1_inst.txt", 'w')
# f_err_lab = open("results/dummy2_inst.txt", 'w')

f_res = open(conf_dict_com["output_folder_name"] + '/' + conf_dict_com["res_filename"], 'a')
metr_dict = init_metr_dict()
for run_ind in range(conf_dict_com["num_runs"]):
	pred_vals, true_vals, metr_dict = train_evaluate_model(data_dict['lab'][:data_dict['train_en_ind']], data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], metr_dict)
metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"])
# f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\n" % (model_type,word_feat_str,sent_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,(metr_dict['avg_fl_ma']+metr_dict['avg_fi'])/2,(metr_dict['std_fl_ma']+metr_dict['std_fi'])/2,metr_dict['avg_fi'],metr_dict['avg_fl_ma'],(metr_dict['avg_fl_ma']+metr_dict['avg_fi']+metr_dict['avg_ji']+metr_dict['avg_fl_mi'])/4,metr_dict['avg_ji'],metr_dict['avg_fl_mi'],metr_dict['avg_em'],metr_dict['avg_ihl'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"]))
write_results(metr_dict, f_res)                                                            
f_res.close()
# f_err_inst.close()
# f_err_lab.close()
# print("prec_macro = %.3f, recall_macro = %.3f" % (pl_ma, rl_ma))





