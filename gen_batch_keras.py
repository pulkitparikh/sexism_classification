import numpy as np
import keras

class TrainGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, word_feats, sent_feats, data_dict, batch_size):
        self.word_feats = word_feats
        self.sent_feats = sent_feats
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        # self.indexes = np.arange(len(list_IDs))

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
        # return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # indexes = self.indexes[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = self.list_IDs[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.list_IDs)

    def __data_generation(self, list_IDs_temp):
        X_inputs = []
        for word_feat in self.word_feats:
            X_inputs.append(np.empty((len(list_IDs_temp), *word_feat['dim_shape'])))

        for i, ID in enumerate(list_IDs_temp):
            input_ind = 0       
            for word_feat in self.word_feats:
                if 'func' in word_feat:
                    X_inputs[input_ind][i,] = word_feat['func'](ID, word_feat, self.data_dict, word_feat['filepath'] + str(ID) + '.npy', word_feat['emb_size'])
                else:
                    X_inputs[input_ind][i,] = np.load(word_feat['filepath'] + str(ID) + '.npy')
                input_ind += 1

        for sent_feat in self.sent_feats:
            X_inputs.append(sent_feat['feats'][list_IDs_temp])

        return X_inputs, self.labels[list_IDs_temp]

class TestGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, word_feats, sent_feats, data_dict, batch_size):
        self.word_feats = word_feats
        self.sent_feats = sent_feats
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.list_IDs = list_IDs

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]

        X = self.__data_generation(list_IDs_temp)
        return X

    def __data_generation(self, list_IDs_temp):
        X_inputs = []
        for word_feat in self.word_feats:
            X_inputs.append(np.empty((len(list_IDs_temp), *word_feat['dim_shape'])))

        for i, ID in enumerate(list_IDs_temp):
            input_ind = 0       
            for word_feat in self.word_feats:
                if 'func' in word_feat:
                    X_inputs[input_ind][i,] = word_feat['func'](ID, word_feat, self.data_dict, word_feat['filepath'] + str(ID) + '.npy', word_feat['emb_size'])
                else:
                    X_inputs[input_ind][i,] = np.load(word_feat['filepath'] + str(ID) + '.npy')
                input_ind += 1

        for sent_feat in self.sent_feats:
            X_inputs.append(sent_feat['feats'][list_IDs_temp])

        return X_inputs