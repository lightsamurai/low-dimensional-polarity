t1 = pickle.load(open("./data/embeddings/random_docs.pkl", "rb"))
t1 = t1["embeddings"]
he_train = np.vstack((t1[:4000], t1[7000:11000]))
he_test = np.vstack((t1[4000:7000], t1[11000: 14000]))
hr_train = np.vstack((t1[:4000], t1[14000:18000]))
hr_test = np.vstack((t1[4000:7000], t1[18000:21000]))
hg_train = np.vstack((t1[:4000], t1[21000:25000]))
hg_test = np.vstack((t1[4000:7000], t1[25000:28000]))
er_train = np.vstack((t1[7000:11000], t1[14000:18000]))
er_test = np.vstack((t1[11000: 14000], t1[18000:21000]))
eg_train = np.vstack((t1[7000:11000], t1[21000:25000]))
eg_test = np.vstack((t1[11000: 14000], t1[25000:28000]))
rg_train = np.vstack((t1[14000:18000], t1[21000:25000]))
rg_test = np.vstack((t1[18000:21000], t1[25000:28000]))



import os
import json
import pickle
from sys import argv
from docopt import docopt
from numpy import concatenate
from sklearn import svm
from sklearn.svm import SVC # C-Support Vector Classification
from utils_svm import *
from utils import *

x_train = he_train ##function from utils to import train and test files? !!!!!!!
x_test  = he_test
y_train = make_labels(4000, 4000)
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
#make_confmat(y_pred, y_test, t1, t2) 

# y = make_labels(4000, 4000)

svm = svm.SVC(kernel='linear')
svm.fit(he_train, y_train)
w = svm.coef_

with open('w_he.pkl', "wb") as fOut: # remember to change the name of the weight vector !!!!!!
    pickle.dump(w, fOut, protocol=pickle.HIGHEST_PROTOCOL)
