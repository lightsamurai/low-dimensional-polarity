# low-dimensional-polarity
Extreme pruning: a proof of concept for a low-dimensional polarity classification model.

The /utils folder contains all the steps of the pipeline.
1_Reddit_scraping.py enables to scrape Reddit comments from a given subreddit.
2_process_scraped_files.py organizes the extracted data in .json files, keeping a certain number of texts from the comments.
3_encode_into_embeddings.py encodes each comment into a 384-dimensional sentence embedding using S-BERT. Then, it organizes all embeddings into a .pkl file. In the end, each .pkl file will contain embeddings extracted in equal number from two polarised subreddits (e.g. prolife vs prochoice, climate vs climate change deniers, republicans vs democrats...).
4_Experiments_1_and_2.py searches single dimensions which are determinant for dividing the embeddings contained in a single .pkl file into its two polarised component. The procedure is applied to the raw embeddings files and to them after having been normalised and having applied PCA on the matrix.
5_svm_polarised.py applies the Support Vector Machines algorithm to binary classify the two classes of polarised documents (embeddings).
5_svm_topics.py applies the Support Vector Machines algorithm to binary classify the two classes of topical documents (embeddings).
6_polarised_w_reduced.py supervisedly searches subsets of the 384 dimensions (i.e. linear combinations of those dimensions, resulting from SVM) which are sufficient for classifying the embeddings into two polarised classes.
6_topical_w_reduced.py supervisedly searches subsets of the 384 dimensions (i.e. linear combinations of those dimensions, resulting from SVM) which are sufficient for classifying the embeddings into two classes, depending on the topic dealt.
utils.py are the utilities necessaries for making the whole pipeline work.
utils_svm.py  are the utilities necessaries for the 5_svm_polarised.py and 5_svm_topics.py files.
