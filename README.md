# sexism_classification
We hereby release the code that was used for the experiments reported in our following EMNLP 2019 paper:

Pulkit Parikh, Harika Abburi, Pinkesh Badjatiya, Radhika Krishnan, Niyati Chhaya, Manish Gupta, and Vasudeva Varma. 2019. Multi-label Categorization of Accounts of Sexism using a Neural Framework. To be published in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*

The code, in fact, provides some additional functionality as well. Our implementation utilizes parts of the code from [1, 2, 3] and libraries Keras and Scikit-learn [4]. The following are brief descriptions of the contents of this repository. Any further information about the code can be obtained by emailing me (my email id is mentioned in the paper).

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

Under the guidance of a social scientist, we created a dataset of 13000+ accounts of sexism annotated with one or more of 23 carefully formulated categories of sexism by ten people, most of whom have formally studied gender and/or sexuality. We are committed to ensuring that data is used and released ethically, which includes protecting the privacy and anonymity of the victims. Our full dataset [can be requested](https://irel.iiit.ac.in/sexism-classification/dataset-request) for academic purposes alone by providing some prerequisites as recommended by an ethics committee and agreeing to certain terms. The requesters who fulfill these preconditions will be given the data in such a way that an account that gets removed from ESP by a victim (or contributor) will no longer be included in the data yielded by our script.

References:

[1] Sweta Agrawal and Amit Awekar. 2018. Deep learning for detecting cyberbullying across multiple social media platforms. In European Conference on Information Retrieval, pages 141–153. Springer.

[2] Nikhil Pattisapu, Manish Gupta, Ponnurangam Kumaraguru, and Vasudeva Varma. 2017. Medical persona classification in social media. In Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining
2017, pages 377–384. ACM.   

[3] Richard Liao. 2017. textclassifier. https://github.com/richliao/textClassifier.

[4] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.
