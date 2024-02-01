from sklearn.metrics import pairwise_distances
import pickle
import numpy as np

# DATA: load W vectors and matrices

w_abo = pickle.load(open("./data/w_vectors_polarised/w_abortion.pkl", "rb"))
w_abo = w_abo[0,:]

w_cli = pickle.load(open("./data/w_vectors_polarised//w_climate.pkl", "rb"))
w_cli = w_cli[0,:]

w_vax = pickle.load(open("./data/w_vectors_polarised/w_vectors/w_vax.pkl", "rb"))
w_vax = w_vax[0,:]

w_pol = pickle.load(open("./data/w_vectors_polarised/w_vectors/w_politics.pkl", "rb"))
w_pol = w_pol[0,:]

# w_all = np.vstack((w_abo, w_cli, w_vax, w_pol))
# print(1- pairwise_distances(w_all, metric="cosine")/2)

# load TRAINING and TEST SETS
abortion_dataset = pickle.load(open("./data/embeddings/all_abortion.pkl",'rb'))
abortion_dataset = abortion_dataset['embeddings']
train_abortion_dataset = np.vstack((abortion_dataset[:7000], abortion_dataset[10000:17000]))
test_abortion_dataset = np.vstack((abortion_dataset[7000:10000], abortion_dataset[17000:20000]))
print("Train abortion_dataset:", train_abortion_dataset.shape, "\n", "Test x:", test_abortion_dataset.shape, "\n")

climate_dataset = pickle.load(open("./data/embeddings/all_climate.pkl",'rb'))
climate_dataset = climate_dataset['embeddings']
train_climate_dataset = np.vstack((climate_dataset[:7000], climate_dataset[10000:17000]))
test_climate_dataset = np.vstack((climate_dataset[7000:10000], climate_dataset[17000:20000]))
print("Train climate_dataset:", train_climate_dataset.shape, "\n", "Test climate_dataset:", test_climate_dataset.shape, "\n")

vax_dataset = pickle.load(open("./data/embeddings/all_vax.pkl",'rb'))
vax_dataset = vax_dataset['embeddings']
train_vax_dataset = np.vstack((vax_dataset[:7000], vax_dataset[10000:17000]))
test_vax_dataset = np.vstack((vax_dataset[7000:10000], vax_dataset[17000:20000]))
print("Train vax_dataset:", train_vax_dataset.shape, "\n", "Test vax_dataset:", test_vax_dataset.shape, "\n")

politics_dataset = pickle.load(open("./data/embeddings/all_politics.pkl",'rb'))
politics_dataset = politics_dataset['embeddings']
train_politics_dataset = np.vstack((politics_dataset[:7000], politics_dataset[10000:17000]))
test_politics_dataset = np.vstack((politics_dataset[7000:10000], politics_dataset[17000:20000]))
print("Train politics_dataset:", train_politics_dataset.shape, "\n", "Test politics_dataset:", test_politics_dataset.shape, "\n")


# sort the dimensions of the w vector from the most important ones
ind_abo = np.flip(np.argsort(abs(w_abo),))
# print(ind_abo[:38])
ind_cli = np.flip(np.argsort(abs(w_cli),))
# print(ind_cli[:38])
ind_vax = np.flip(np.argsort(abs(w_vax),))
# print(ind_vax[:38])
ind_pol = np.flip(np.argsort(abs(w_pol),))
# print(ind_pol[:38])

# among the first 128, find the common ones: they're 9
reduced_ = list(set(ind_abo[:128]) & set(ind_cli[:128]) & set(ind_vax[:128])  &set(ind_pol[:128]))
print("Common dimensions:", reduced_) # subset of common dimensions

#IMPORTANT: for the leave-one-out procedure subtract one of the datasets:

# find the averaged w vector leaving politics out
red = np.array([w_abo[reduced_], w_cli[reduced_], w_vax[reduced_]])


w_mean = np.mean(red,axis=0) # mean of all four dimensions
# print(w_mean)

# testing our procedure on the left-out dataset


scores = []
for i in range(6000):
    val = w_mean.dot(test_politics_dataset[i][reduced_]) #testing with only common dims as indices
    if val < 0:
        scores.append(1)
    else:
        scores.append(0)
    print("Accuracy on left-out test set:", sum(scores)/6000)

scores = []
for i in range(6000):
    val = w_mean.dot(test_politics_dataset[i][reduced_]) #test set with only common dims as indices
    if val > 0:
        scores.append(1)
    else:
        scores.append(0)
    print("Accuracy on left-out test set:", sum(scores)/6000)

