import tensorflow as tf
from keras import backend as K
from keras.layers import TimeDistributed, Embedding, Dense, Input, Flatten, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU, Bidirectional, concatenate
from keras.models import Model
from keras import optimizers
from keras.engine.topology import Layer
from keras import initializers

def sen_embed(enc_algo, sen_word_emb, word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs):
    if enc_algo == "rnn":
        if att_dim > 0:
            rnn_sen_mod, att_mod = rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
            rnn_sen_emb_output = TimeDistributed(rnn_sen_mod)(sen_word_emb)    
            att_outputs.append(TimeDistributed(att_mod)(sen_word_emb))
            # rnn_sen_mod.summary()
        else:
            rnn_sen_emb_output = TimeDistributed(rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type))(sen_word_emb)    
        return [rnn_sen_emb_output], att_outputs
    elif enc_algo == "cnn":
        cnn_sen_mod = cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes) 
        # cnn_sen_mod.summary()
        return [TimeDistributed(cnn_sen_mod)(sen_word_emb)], att_outputs
    else:
        if att_dim > 0:
            rnn_sen_mod, att_mod = rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
            rnn_sen_emb_output = TimeDistributed(rnn_sen_mod)(sen_word_emb)    
            att_outputs.append(TimeDistributed(att_mod)(sen_word_emb))
        else:
            rnn_sen_emb_output = TimeDistributed(rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type))(sen_word_emb)    
        cnn_sen_emb_output = TimeDistributed(cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes))(sen_word_emb)
        
        if enc_algo == "comb_cnn_rnn":
            return [concatenate([cnn_sen_emb_output, rnn_sen_emb_output])], att_outputs
        elif enc_algo == "sep_cnn_rnn":
            return [cnn_sen_emb_output, rnn_sen_emb_output], att_outputs

def flat_embed(enc_algo, word_emb_seq, word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs):
    if enc_algo == "rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
        # rnn_mod.summary()
        return rnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "cnn":
        cnn_mod = cnn_sen_embed(word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes)
        # cnn_mod.summary()
        return cnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "comb_cnn_rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
            rnn_emb_output = rnn_mod(word_emb_seq)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_emb_output = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)(word_emb_seq)
        cnn_emb_output = cnn_sen_embed(word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes)(word_emb_seq)

        return concatenate([cnn_emb_output, rnn_emb_output]), att_outputs

def rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(w_emb_input_seq)
    else:
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(w_emb_input_seq)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
        return Model(w_emb_input_seq, blstm_l), Model(w_emb_input_seq, att_w)
    else:
        return Model(w_emb_input_seq, blstm_l)

def cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(w_emb_input_seq)
        if max_pool_k_val == 1:
            pool_t = GlobalMaxPooling1D()(conv_t)
        else:
            pool_t = kmax_pooling(max_pool_k_val)(conv_t)
        conv_l_list.append(pool_t)
    feat_vec = concatenate(conv_l_list)
    return Model(w_emb_input_seq, feat_vec)

def post_embed(sen_emb, rnn_dim, att_dim, rnn_type, stack_rnn_flag, att_outputs):
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(sen_emb)
        if stack_rnn_flag:
            blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(blstm_l)
    else:
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(sen_emb)
        if stack_rnn_flag:
            blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(blstm_l)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
        att_outputs.append(att_w)
    return blstm_l, att_outputs

def add_word_sen_emb_p1(model_inputs, word_emb_input, word_f_sen_word_emb, word_f_sen_emb_size, enc_algo, stage1_id, stage2_id, p1_dict):
        model_inputs.append(word_emb_input)
        if stage1_id in p1_dict:
            p1_dict[stage1_id]["comb_feature_list"].append(word_f_sen_word_emb)
            p1_dict[stage1_id]["word_emb_len"] += word_f_sen_emb_size
            p1_dict[stage1_id]["enc_algo"] = enc_algo
        else:
            p1_dict[stage1_id] = {}
            p1_dict[stage1_id]["comb_feature_list"] = [word_f_sen_word_emb]
            p1_dict[stage1_id]["word_emb_len"] = word_f_sen_emb_size
            p1_dict[stage1_id]["stage2"] = stage2_id 
            p1_dict[stage1_id]["enc_algo"] = enc_algo

def add_sen_emb_p2(sen_emb, stage2_id, p2_dict):
        if stage2_id in p2_dict:
            p2_dict[stage2_id]["comb_feature_list"].append(sen_emb)
        else:
            p2_dict[stage2_id] = {}
            p2_dict[stage2_id]["comb_feature_list"] = [sen_emb]

def add_word_emb_p_flat(model_inputs, word_emb_input, word_f_word_emb, word_f_emb_size, enc_algo, m_id, p_dict):
        model_inputs.append(word_emb_input)
        if m_id in p_dict:
            p_dict[m_id]["comb_feature_list"].append(word_f_word_emb)
            p_dict[m_id]["word_emb_len"] += word_f_emb_size
            p_dict[m_id]["enc_algo"] = enc_algo
        else:
            p_dict[m_id] = {}
            p_dict[m_id]["comb_feature_list"] = [word_f_word_emb]
            p_dict[m_id]["word_emb_len"] = word_f_emb_size 
            p_dict[m_id]["enc_algo"] = enc_algo

def hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes):
    p1_dict = {}
    p2_dict = {}
    model_inputs = []
    att_outputs = []

    for word_feat in word_feats:
        if 'embed_mat' in word_feat:
            word_f_input, word_f_sen_word_emb = tunable_embed_hier_embed(sent_cnt, word_cnt_sent, len(word_feat['embed_mat']), word_feat['embed_mat'], word_feat['emb'], dropO1)
            add_word_sen_emb_p1(model_inputs, word_f_input, word_f_sen_word_emb, word_feat['embed_mat'].shape[-1], word_feat['s_enc'], word_feat['m_id'][0], word_feat['m_id'][1:], p1_dict)
        else:
            word_f_input = Input(shape=(sent_cnt, word_cnt_sent, word_feat['dim_shape'][-1]), name=word_feat['emb'])
            word_f_sen_word_emb = Dropout(dropO1)(word_f_input)
            add_word_sen_emb_p1(model_inputs, word_f_input, word_f_sen_word_emb, word_feat['dim_shape'][-1], word_feat['s_enc'], word_feat['m_id'][0], word_feat['m_id'][1:], p1_dict)
    
    for my_dict in p1_dict.values():
        my_dict["sen_word_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        sen_emb_list, att_outputs = sen_embed(my_dict["enc_algo"], my_dict["sen_word_emb"], word_cnt_sent, my_dict["word_emb_len"], dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs)
        for ind, sen_emb in enumerate(sen_emb_list):
            add_sen_emb_p2(sen_emb, my_dict["stage2"][ind], p2_dict)

    for sen_feat in sen_feats:
        sen_f_input = Input(shape=(sent_cnt, sen_feat['feats'].shape[-1]), name=sen_feat['emb'])
        model_inputs.append(sen_f_input)        
        sen_f_dr1 = Dropout(dropO1)(sen_f_input)
        add_sen_emb_p2(sen_f_dr1, sen_feat['m_id'], p2_dict)

    post_vec_list = []    
    for stage2_val, my_dict in p2_dict.items():
        my_dict["sen_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        post_emb, att_outputs = post_embed(my_dict["sen_emb"], rnn_dim, att_dim, rnn_type, stack_rnn_flag, att_outputs)
        post_vec_list.append(post_emb)
    post_vec = concatenate(post_vec_list) if len(post_vec_list) > 1 else post_vec_list[0]
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    att_mod = Model(model_inputs, att_outputs) if att_outputs else None
    return apply_dense(model_inputs, dropO2, post_vec, nonlin, out_vec_size), att_mod

def flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes):
    p_dict = {}
    model_inputs = []
    att_outputs = []

    for word_feat in word_feats:
        if 'embed_mat' in word_feat:
            word_f_input, word_f_word_emb_raw = tunable_embed_apply(word_cnt_post, len(word_feat['embed_mat']), word_feat['embed_mat'], word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_word_emb_raw)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['embed_mat'].shape[-1], word_feat['s_enc'], word_feat['m_id'], p_dict)
        else:
            word_f_input = Input(shape=(word_cnt_post, word_feat['dim_shape'][-1]), name=word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_input)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['dim_shape'][-1], word_feat['s_enc'], word_feat['m_id'], p_dict)
    
    post_vec_list = []    
    for my_dict in p_dict.values():
        my_dict["word_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        flat_emb, att_outputs = flat_embed(my_dict["enc_algo"], my_dict["word_emb"], word_cnt_post, my_dict["word_emb_len"], dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs)
        post_vec_list.append(flat_emb)
    post_vec = concatenate(post_vec_list) if len(post_vec_list) > 1 else post_vec_list[0]
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    att_mod = Model(model_inputs, att_outputs) if att_outputs else None
    return apply_dense(model_inputs, dropO2, post_vec, nonlin, out_vec_size), att_mod

def apply_dense(input_seq, dropO2, post_vec, nonlin, out_vec_size):
    dr2_l = Dropout(dropO2)(post_vec)
    out_vec = Dense(out_vec_size, activation=nonlin)(dr2_l)
    return Model(input_seq, out_vec)

def c_bilstm(word_cnt_post, word_f, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size, rnn_type, num_cnn_filters, kernel_sizes):
    if 'embed_mat' in word_f:
        input_seq, embedded_seq = tunable_embed_apply(word_cnt_post, len(word_f['embed_mat']), word_f['embed_mat'])
        dr1_l = Dropout(dropO1)(embedded_seq)
    else:
        input_seq = Input(shape=(word_cnt_post, word_f['dim_shape'][-1]))
        dr1_l = Dropout(dropO1)(input_seq)

    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(dr1_l)
        conv_l_list.append(conv_t)
    conc_mat = concatenate(conv_l_list)
    return rnn_dense_apply(conc_mat, input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type), None

def uni_sent(sent_cnt, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size, rnn_type, given_sen_feat):
    aux_input_seq = Input(shape=(sent_cnt, given_sen_feat['feats'].shape[-1]), name='sen_input')
    aux_dr1 = Dropout(dropO1)(aux_input_seq)
    return rnn_dense_apply(aux_dr1, aux_input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type), None

def rnn_dense_apply(rnn_seq, input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type):
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(rnn_seq)
    else:
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(rnn_seq)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
    return apply_dense(input_seq, dropO2, blstm_l, nonlin, out_vec_size)

def tunable_embed_apply(word_cnt_post, vocab_size, embed_mat, word_feat_name):
    input_seq = Input(shape=(word_cnt_post,), name=word_feat_name+'_t')
    embed_layer = Embedding(vocab_size, embed_mat.shape[1], embeddings_initializer=initializers.Constant(embed_mat), input_length=word_cnt_post, name=word_feat_name)
    embed_layer.trainable = True
    embed_l = embed_layer(input_seq)
    return input_seq, embed_l

def tunable_embed_hier_embed(sent_cnt, word_cnt_sent, vocab_size, embed_mat, word_feat_name, dropO1):
    word_input_seq, embed_l = tunable_embed_apply(word_cnt_sent, vocab_size, embed_mat, word_feat_name)
    emb_model = Model(word_input_seq, embed_l)
    sen_input_seq = Input(shape=(sent_cnt, word_cnt_sent), name=word_feat_name+'_t')
    time_l = TimeDistributed(emb_model)(sen_input_seq)
    dr1_l = Dropout(dropO1)(time_l)
    return sen_input_seq, dr1_l

# adapted from https://github.com/richliao/textClassifier
class attLayer_hier(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('glorot_uniform')
        # self.init = initializers.get('normal')
        # self.supports_masking = True
        self.attention_dim = attention_dim
        super(attLayer_hier, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert len(input_shape) == 3
        self.W = self.add_weight(name = 'W', shape = (input_shape[-1], self.attention_dim), initializer=self.init, trainable=True)
        self.b = self.add_weight(name = 'b', shape = (self.attention_dim, ), initializer=self.init, trainable=True)
        self.u = self.add_weight(name = 'u', shape = (self.attention_dim, 1), initializer=self.init, trainable=True)
        super(attLayer_hier, self).build(input_shape)

    # def compute_mask(self, inputs, mask=None):
    #     return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        exp_ait = K.expand_dims(ait)
        weighted_input = x * exp_ait
        output = K.sum(weighted_input, axis=1)
        # print(K.int_shape(ait))

        return [output, ait]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]

    def get_config(self):
        config = {'attention_dim': self.attention_dim}
        base_config = super(attLayer_hier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class kmax_pooling(Layer):
    def __init__(self, k_val, **kwargs):
        self.k_val = k_val
        super(kmax_pooling, self).__init__(**kwargs)

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k_var = tf.nn.top_k(shifted_input, k=self.k_val, sorted=True, name=None)[0]
        
        # return flattened output
        return tf.reshape(top_k_var, [tf.shape(top_k_var)[0], -1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]* self.k_val)

    def get_config(self):
        config = {'k_val': self.k_val}
        base_config = super(kmax_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def lp_categ_loss(weights):
    def lp_categ_of(y_true, y_pred):
        # return K.squeeze(K.dot(y_true, K.expand_dims(K.variable(weights))), -1)
        return K.sum(weights*y_true,axis=1)*K.categorical_crossentropy(y_true, y_pred)
    return lp_categ_of

def br_binary_loss(weights):
    def br_binary_of(y_true, y_pred):
        return ((weights[0]*(1-y_true))+(weights[1]*y_true))*K.binary_crossentropy(y_true, y_pred)
        # return weights[y_true]*K.binary_crossentropy(y_true, y_pred)
    return br_binary_of

def multi_binary_loss(weights):
    def multi_binary_of(y_true, y_pred):
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1)
        # return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
        # return K.mean(weights[cl_arr, y_true]*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return multi_binary_of

def multi_cat_w_loss(weights):
    def multi_cat_w_of(y_true, y_pred):
        return -K.sum(weights*y_true*K.log(y_pred),axis = 1)/K.sum(y_true,axis = 1)
    return multi_cat_w_of

def multi_cat_loss():
    def multi_cat_of(y_true, y_pred):
        return -K.sum(y_true*K.log(y_pred),axis = 1)/K.sum(y_true,axis = 1)
    return multi_cat_of

def get_model(m_type, word_cnt_post, sent_cnt, word_cnt_sent, word_feats, sen_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, kernel_sizes):
    if m_type == 'hier_fuse':
        model, att_mod = hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes)
    elif m_type == 'flat_fuse':
        model, att_mod = flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes)
    elif m_type == 'c_bilstm':
        model, att_mod = c_bilstm(word_cnt_post, list(word_feats.values())[0], rnn_dim, 0, dropO1, dropO2, nonlin, out_vec_size, rnn_type, num_cnn_filters, kernel_sizes)
    elif m_type == 'uni_sent':
        model, att_mod = uni_sent(sent_cnt, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size, rnn_type, list(sen_feats.values())[0])
    else:
        print("ERROR: No model named %s" % m_type)
        return None, None

    adam = optimizers.Adam(lr=learn_rate)
    model.compile(loss=loss_func, optimizer=adam)
    # model.summary()
    # print(att_mod.output_shape)
    return model, att_mod


# import numpy as np
# # # word_feats = {'glove': {'embed_mat': np.zeros((20000, 300)), 'tune': True, 's_enc': 'cnn_rnn_sep', 'emb_size': 300, 'm_id': '113'}, 'elmo': {'s_enc': 'rnn', 'emb_size': 3072, 'm_id': '32'}, 'ling': {'s_enc': 'cnn', 'emb_size': 33, 'm_id': '22'}}
# # # sen_feats = {'infersent': {'emb_size': 4096, 'm_id': '1'}, 'use': {'emb_size': 512, 'm_id': '3'}}

# word_feats = {'elmo': {'s_enc': 'cnn', 'm_id': '11', 'dim_shape': [100,300]}}#, 'glove': {'s_enc': 'rnn', 'm_id': '22', 'dim_shape': [100,300]}}
# sen_feats = {}#'use': {'m_id': '3', 'feats': np.zeros((100, 512))}, 'bert': {'m_id': '1', 'feats': np.zeros((100, 768))}}
# get_model('c_bilstm', 200, 20, 100, word_feats, sen_feats, 0.001, 0.1, 0.1, 80, 'lstm', 'binary_crossentropy', 'sigmoid', 14, 200, 0, 1, False, [2,3,4])
