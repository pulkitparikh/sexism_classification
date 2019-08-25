import numpy as np
from sklearn.model_selection import train_test_split
from loadPreProc import *
from evalMeasures import *
import sys

conf_dict_list, conf_dict_com = load_config(sys.argv[1])

prob_trans_type = "None"
model_type = "random"
vector_type = "None"
embed_size = "None"

def train_evaluate_model(trainY_list, true_vals, metr_dict):
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in trainY_list:
		for l in lset:
			train_coverage[l] += 1.0
	train_coverage /= float(len(trainY_list))

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

	pred_vals = di_op_to_label_lists(r_op)

	return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict)

data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"])

f_res = open(conf_dict_com["output_folder_name"] + '/' + conf_dict_com["res_filename"], 'a')
metr_dict = init_metr_dict()
for run_ind in range(conf_dict_com["num_runs"]):
	pred_vals, true_vals, metr_dict = train_evaluate_model(data_dict['lab'][:data_dict['train_en_ind']], data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], metr_dict)
metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"])
write_results(metr_dict, f_res)                                                            
f_res.close()




