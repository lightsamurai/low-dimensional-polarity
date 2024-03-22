# low-dimensional-polarity
Extreme pruning: a proof of concept for a low-dimensional polarity classification model.

The /utils folder contains all the steps of the pipeline.

**1_Reddit_scraping.py** scrapes Reddit comments from a given subreddit.

**2_process_scraped_files.py** organizes the extracted data in .json files, keeping a certain number of the elements "comments".

**3_encode_into_embeddings.py** encodes each comment into a 384-dimensional sentence embedding using S-BERT. Then, it organizes all embeddings into a .pkl file. In the end, each .pkl file will contain embeddings extracted in equal number from two polarised subreddits (e.g. r/prolife *vs* r/prochoice, r/climate *vs* r/climate change deniers, r/republicans *vs* r/democrats...) or from two topically distinct subreddits (e.g. r/gardening *vs* r/ethereum...).

**4_Experiments_1_and_2.py** searches single dimensions which are determinant for dividing the embeddings contained in a single .pkl file into its two polarised or topically distinct components. The procedure is first applied to the embeddings files, and then after normalising them and applying Principal Component Analysis (PCA) to them.

**5_svm_polarised.py** applies the Support Vector Machines (SVM) algorithm to binary classify the two classes of polarised documents (embeddings).

**5_svm_topics.py** applies the Support Vector Machines (SVM) algorithm to binary classify the two classes of topical documents (embeddings).

**6_polarised_w_reduced.py** supervisedly searches subsets of all 384 dimensions (i.e. linear combinations of those dimensions, resulting from SVM), such as they are sufficient for accurately classifying the embeddings into two polarised classes, depending on the polarity expressed.

**6_topical_w_reduced.py** supervisedly searches subsets of the 384 dimensions (i.e. linear combinations of those dimensions, resulting from SVM), such as they are sufficient for accurately classifying the embeddings into two classes, depending on the topic dealt.

**utils.py** are the utilities for making the whole pipeline work.

**utils_svm.py** are the utilities for the 5_svm_polarised.py and 5_svm_topics.py files.

