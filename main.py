import os
import time
import sys
from sent_enc_embed import sent_enc_featurize
from word_embed import word_featurize
from neuralApproaches import *

sys.setrecursionlimit(10000)
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict_com['GPU_ID']
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)
res_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_filename"]
if os.path.isfile(res_path):
    f_res = open(res_path, 'a')
else:
    f_res = open(res_path, 'w')

tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    f_tsv.write("model\tword feats\tsent feats\ttrans\tclass imb\tcnn fils\tcnn kernerls\tthresh\trnn dim\tatt dim\tpool k\tstack RNN\tf_I+f_Ma\tstd_d\tf1-Inst\tf1-Macro\tsum_4\tJaccard\tf1-Micro\tExact\tI-Ham\trnn type\tl rate\tb size\tdr1\tdr2\ttest mode\n") 

data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"])
print("max # sentences: %d, max # words per sentence: %d, max # words per post: %d" % (data_dict['max_num_sent'], data_dict['max_words_sent'], data_dict['max_post_length']))

metr_dict = init_metr_dict()
for conf_dict in conf_dict_list:
    for prob_trans_type in conf_dict["prob_trans_types"]:
        trainY_list, trainY_noncat_list, num_classes_var, bac_map = transform_labels(data_dict['lab'][:data_dict['train_en_ind']], prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"])
        for class_imb_flag in conf_dict["class_imb_flags"]:
            loss_func_list, nonlin, out_vec_size, cw_list = class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_var, prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"])
            for model_type in conf_dict["model_types"]:
                for word_feats_raw in conf_dict["word_feats_l"]:
                    word_feats, word_feat_str = word_featurize(word_feats_raw, model_type, data_dict, conf_dict_com['poss_word_feats_emb_dict'], conf_dict_com['use_saved_word_feats'], conf_dict_com['save_word_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"])
                    for sent_enc_feats_raw in conf_dict["sent_enc_feats_l"]:
                        sent_enc_feats, sent_enc_feat_str = sent_enc_featurize(sent_enc_feats_raw, model_type, data_dict, conf_dict_com['poss_sent_enc_feats_emb_dict'], conf_dict_com['use_saved_sent_enc_feats'], conf_dict_com['save_sent_enc_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"])
                        for num_cnn_filters in conf_dict["num_cnn_filters"]:
                            for max_pool_k_val in conf_dict["max_pool_k_vals"]:
                                for cnn_kernel_set in conf_dict["cnn_kernel_sets"]:
                                    cnn_kernel_set_str = str(cnn_kernel_set)[1:-1].replace(',','').replace(' ', '')
                                    for rnn_type in conf_dict["rnn_types"]:
                                        for rnn_dim in conf_dict["rnn_dims"]:
                                            for att_dim in conf_dict["att_dims"]:
                                                for stack_rnn_flag in conf_dict["stack_rnn_flags"]:
                                                    mod_op_list_save_list = []
                                                    for thresh in conf_dict["threshes"]:
                                                        startTime = time.time()
                                                        info_str = "model: %s, word_feats = %s, sent_enc_feats = %s, prob_trans_type = %s, class_imb_flag = %s, num_cnn_filters = %s, cnn_kernel_set = %s, rnn_type = %s, rnn_dim = %s, att_dim = %s, max_pool_k_val = %s, stack_rnn_flag = %s, thresh = %s, test mode = %s" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, thresh, conf_dict_com["test_mode"])
                                                        fname_part = ("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, conf_dict_com["test_mode"]))
                                                        for run_ind in range(conf_dict_com["num_runs"]):
                                                            print('run: %s; %s\n' % (run_ind, info_str))
                                                            if run_ind < len(mod_op_list_save_list):
                                                                mod_op_list = mod_op_list_save_list[run_ind]   
                                                            else:
                                                                mod_op_list = []
                                                                for m_ind, (loss_func, cw, trainY) in enumerate(zip(loss_func_list, cw_list, trainY_list)):
                                                                    mod_op, att_op = train_predict(word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, m_ind, run_ind, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"])
                                                                    mod_op_list.append((mod_op, att_op))
                                                                mod_op_list_save_list.append(mod_op_list) 
                                                            pred_vals, true_vals, metr_dict = evaluate_model(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, thresh, conf_dict_com['gen_att'], conf_dict_com["output_folder_name"], ("%s~%d" % (fname_part,run_ind)))
                                                            if conf_dict_com['gen_inst_res'] and run_ind == 0:
                                                                insights_results(pred_vals, true_vals, data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']], data_dict['text_sen'][data_dict['test_st_ind']:data_dict['test_en_ind']], data_dict['lab'][0:data_dict['train_en_ind']], fname_part, conf_dict_com["output_folder_name"])
                                                        f_res.write("%s\n\n" % info_str)
                                                        print("%s\n" % info_str)
                                                        metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"])
                                                        f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\n" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,(metr_dict['avg_fl_ma']+metr_dict['avg_fi'])/2,(metr_dict['std_fl_ma']+metr_dict['std_fi'])/2,metr_dict['avg_fi'],metr_dict['avg_fl_ma'],(metr_dict['avg_fl_ma']+metr_dict['avg_fi']+metr_dict['avg_ji']+metr_dict['avg_fl_mi'])/4,metr_dict['avg_ji'],metr_dict['avg_fl_mi'],metr_dict['avg_em'],metr_dict['avg_ihl'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"]))
                                                        write_results(metr_dict, f_res)                                                            
                                                        timeLapsed = int(time.time() - startTime + 0.5)
                                                        hrs = timeLapsed/3600.
                                                        t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
                                                        print(t_str)                
                                                        f_res.write("%s\n" % t_str)

f_res.close()
f_tsv.close()




