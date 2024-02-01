# TOPICAL VECTORS

from sklearn.metrics import pairwise_distances
import pickle
import numpy as np

# DATA: load W vectors and matrices

w_he = pickle.load(open("./data/w_vectors_topics/w_he.pkl", "rb"))
w_he = w_he[0]

w_hr = pickle.load(open("./data/w_vectors_topics/w_hr.pkl", "rb"))
w_hr = w_hr[0]

w_hg = pickle.load(open("./data/w_vectors_topics/w_hg.pkl", "rb"))
w_hg = w_hg[0]

w_er = pickle.load(open("./data/w_vectors_topics/w_er.pkl", "rb"))
w_er = w_er[0]

w_eg = pickle.load(open("./data/w_vectors_topics/w_eg.pkl", "rb"))
w_eg = w_eg[0]

w_rg = pickle.load(open("./data/w_vectors_topics/w_rg.pkl", "rb"))
w_rg = w_rg[0]


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


# sort the dimensions of the w vector
ind_he = np.flip(np.argsort(abs(w_he),))
# print(ind_he[:128])
ind_hr = np.flip(np.argsort(abs(w_hr),))
# print(ind_hr[:128])
ind_hg = np.flip(np.argsort(abs(w_hg),))
# print(ind_hg[:128])
ind_er = np.flip(np.argsort(abs(w_er),))
# print(ind_er[:128])
ind_eg = np.flip(np.argsort(abs(w_eg),))
# print(ind_eg[:128])
ind_rg = np.flip(np.argsort(abs(w_rg),))
# print(ind_gr[:128])


reduced_ = list( set(set(ind_he[:128]) & set(ind_hr[:128]) &set(ind_er[:128]) ))
#print(reduced_) # subset of common dimensions

# compute the similarity leaving recipes out
red = np.array([w_he[reduced_], w_hr[reduced_], w_er[reduced_]])
#print(1- pairwise_distances(red, metric="cosine")/2)

# compute the mean of the weight vector 
w_mean = np.mean(red,axis=0) # mean of all four dimensions
# print(w_mean)

scores = []
for i in range(6000):
    val = w_mean.dot(rg_test[i][reduced_]) #testing with only common dims as indices
    if val < 0:
        scores.append(1)
    else:
        scores.append(0)
    print("Accuracy on left-out test set:", sum(scores)/6000)

scores = []
for i in range(6000):
    val = w_mean.dot(rg_test[i][reduced_]) #test set with only common dims as indices
    if val > 0:
        scores.append(1)
    else:
        scores.append(0)
    print("Accuracy on left-out test set:", sum(scores)/6000)
