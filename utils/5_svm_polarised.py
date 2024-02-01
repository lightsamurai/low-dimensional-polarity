import os
import json
import pickle
from sys import argv
from docopt import docopt
from numpy import concatenate
from sklearn import svm
from sklearn.svm import SVC # C-Support Vector Classification
from utils_svm import *

t1 = pickle.load(open("./data/all_abortion.pkl",'rb'))
# t1 = t1["embeddings"]
t1 = t1[:10000]

t2 = pickle.load(open("./data/all_abortion.pkl",'rb'))
# t2 = t2["embeddings"]
t2 = t2[-10000:]

t1_train = t1[:7000]
# t1_train = t1_train["embeddings"]

t1_test = t1[7000:10000]
# t1_test = t1_test["embeddings"]

t1_train_docs = [json.loads(line) for line in open("./data/texts/all_bodies_abortion.json", 'r')] # should be prolife
t1_train_docs = [comment for sublist in t1_train_docs for comment in sublist]
t1_train_docs = t1_train_docs[:7000]

t1_test_docs = [json.loads(line) for line in open("./data/texts/all_bodies_abortion.json", 'r')] # should be prolife
t1_test_docs = [comment for sublist in t1_test_docs for comment in sublist]
t1_test_docs = t1_test_docs[7000:10000]


t2_train = t2[:7000]
# t2_train = t2_train["embeddings"]

t2_test = t2[7000:10000]
# t2_test = t2_test["embeddings"]


t2_train_docs = [json.loads(line) for line in open("./data/texts/all_bodies_abortion.json", 'r')]
t2_train_docs = [comment for sublist in t2_train_docs for comment in sublist]
t2_train_docs = t2_train_docs[:7000]

t2_test_docs = [json.loads(line) for line in open("./data/texts/all_bodies_abortion.json", 'r')]
t2_test_docs = [comment for sublist in t2_test_docs for comment in sublist]
t2_test_docs = t2_test_docs[7000:10000]

train_docs = t1_train_docs + t2_train_docs

test1_size = len(t1_test)
test2_size = len(t2_test)

print('Topic 1: Train size: {} | Test size: {}\n' .format(7000, 3000) + \
	'Topic 2: Train size: {} | Test size: {}\n' .format(7000, 3000))

	
#prepare train/test sets

x_train = concatenate([t1_train, t2_train])
x_test  = concatenate([t1_test,  t2_test])
y_train = make_labels(7000, 7000)
y_test  = make_labels(3000, 3000)

#setup SVM setup and print output
#Soft Margin SVM can be implemented in Scikit-Learn by adding a C penalty term in svm.SVC
#the kernel function acts as a modified dot product

print('SVC output:')

clf    = SVC(C = 1.0, verbose = True, kernel = "linear") #prints data
model  = clf.fit(x_train, y_train)
score  = clf.score(x_test, y_test)
y_pred = clf.predict(x_test)

print('\n') #needed because SVC prints output in a weird way
print('SVC Model:')
print(model)
print()

print('Score: {}\n'.format(score))

#print('Training docs:')
#print('\n'.join([train_docs[s] for s in clf.support_]))
#print()

#make confusion matrix
make_confmat(y_pred, y_test, t1, t2)

# save weight vector coefficients

svm = svm.SVC(kernel='linear')
svm.fit(t2_train, y_train)
w = svm.coef_

import pickle
with open('weight_vector.pkl', "wb") as fOut:
    pickle.dump(w, fOut, protocol=pickle.HIGHEST_PROTOCOL)


