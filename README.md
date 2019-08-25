# sexism_classification
We hereby release the code that was used for the experiments reported in our EMNLP 2019 submission. The code, in fact, provides some additional functionality as well. We have anonymized paths and other things that could reveal the identity of the authors. Any further information about the code can be obtained by emailing the authors. The following are brief descriptions of the contents of this code directory.

1) main.py

- The main file that needs to be run for all deep learning based methods including the proposed approach and baselines

2) neural_approaches.py

- Training, prediction, evaluation, training data creation/transformation, loss function assignment, class imbalance correction

3) dl_models.py

- Deep learning architectures for the proposed approach as well as baselines

4) load_preproc.py

- Data loading, pre-procressing and other utilities

5) sent_enc_embed.py

- Generation of sentence representations using general-purpose sentence encoding schemes

6) word_embed.py

- Generation of distributional word representations

7) ling_word_feats.py 

- Generation of a linguistic/aspect-based word-level representation

8) gen_batch_keras.py 

- Generation of batches of inputs for training and testing

9) bert_pretraining.py

- Functions related to the pre-training of BERT on a domain-specific corpus (esp. around data creation) 

10) eval_measures.py 

- Functions related to multi-label evaluation and result reporting

11) traditional_ML_LP.py 

- Traditional machine learning methods on ngram based and other features

12) doc2vec_embed.py 

- Creation of a vector representation of a piece of text using doc2vec 

13) rand_approach.py 

- Random label assignment in accordance with normalized training frequencies of labels

14) rand_sample.py

- Creation of a small random sample of the data for fast experimentation

15) config_deep_learning.txt

- Configuration file for deep learning methods specifying multiple nested and non-nested parameter combinations

16) config_traditional_ML.txt

- Configuration file for traditional machine learning methods

