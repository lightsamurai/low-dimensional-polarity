import pickle
import numpy as np

#1 - CLEAN 20K MATRICES
x = pickle.load(open("./data/embeddings/all_abortion.pkl",'rb'))
x = x['embeddings']
print("Abortion original matrix x:", x.shape)

v = pickle.load(open("./data/embeddings/all_climate.pkl",'rb'))
v = v['embeddings']
print("Climate original matrix v:", v.shape)

z = pickle.load(open("./data/embeddings/all_vax.pkl",'rb'))
z = z['embeddings']
print("Vaccine original matrix z:", z.shape)

w = pickle.load(open("./data/embeddings/all_politics.pkl",'rb'))
w = w['embeddings']
print("Politics original matrix w:", w.shape, "\n")

#2 - standardize and save the 4 original matrices standardized

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_scal=scaler.fit_transform(x) # abortion
# print("Abortion standardized matrix X_scal:", X_scal.shape)
V_scal=scaler.fit_transform(v) # climate
# print("Climate standardized matrix V_scal:", V_scal.shape)
Z_scal=scaler.fit_transform(z) # vaccine
# print("Vaccine standardized matrix Z_scal:", Z_scal.shape)
W_scal=scaler.fit_transform(w) # politics
# print("Politics standardized matrix W_scal:", W_scal.shape, "\n")

#3 - computing PCA matrices

from sklearn.decomposition import PCA
pca = PCA(384) # 384 components

X_pca=pca.fit_transform(X_scal)
V_pca=pca.fit_transform(V_scal)
Z_pca=pca.fit_transform(Z_scal)
W_pca=pca.fit_transform(W_scal)

with open('pca_abortion.pkl', "wb") as fOut:
    pickle.dump(X_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_climate.pkl', "wb") as fOut:
    pickle.dump(V_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_vaccine.pkl', "wb") as fOut:
    pickle.dump(Z_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_politics.pkl', "wb") as fOut:
    pickle.dump(W_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#2 - load PCA MATRICES

U = pickle.load(open("./pca_abortion.pkl",'rb'))
print("Abortion pca U matrix:", U.shape)
J = pickle.load(open("./pca_climate.pkl",'rb'))
print("Climate pca J matrix:", J.shape)
F = pickle.load(open("./pca_vaccine.pkl",'rb'))
print("Vaccine pca F matrix:", F.shape)
C = pickle.load(open("./pca_politics.pkl",'rb'))
print("Politics pca C matrix:", C.shape, "\n")

# Experiment 1 and 2: sorting tuples and evaluating dimensions for POLARISED dataset, original matrix and PCA matrix

def sort_tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
         
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j]= tup[j + 1]
                tup[j + 1]= temp
    return tup

def evaluate_dimensions(x):
    scores = []
    for col in x.T: # trasposed matrix: docs in the rows, dims in the col
        mn = np.mean(col)
        c1h = len(np.where(col[:10000] > mn)[0]) # look at the records in the first cluster that are above the mean
        scores.append(abs(0.5 - c1h / 10000)) 
    #print('SCORES:',scores)
    ind = np.argpartition(scores, -5)[-5:]
    best = [(i,scores[i]) for i in ind]
    sort = sort_tuple(best)   
    print('BEST DIM:', sort[-1:])
    
    best_indices = [i[0] for i in sort] 
    # print("BEST indices:", best_indices)


# original matrices    
evaluate_dimensions(x) # abortion 
evaluate_dimensions(v) # climate
evaluate_dimensions(z) # vaccine
evaluate_dimensions(w) # politics


# pca matrices

evaluate_dimensions(U) # abortion
evaluate_dimensions(J) # climate
evaluate_dimensions(F) # vaccine
evaluate_dimensions(C) # politics




# Experiment 1 and 2: sorting tuples and evaluating dimensions for TOPICAL dataset, original matrix and PCA matrix

#2 - CONTROL CONDITIONS DATASETS

rand = pickle.load(open("./data/embeddings/random_docs.pkl", "rb"))
rand = rand["embeddings"]

hiking = rand[:7000]
hiking_train = rand[:4000]
hiking_test = rand[4000:7000]

ethereum = rand[7000:14000]
ethereum_train = rand[7000:11000]
ethereum_test = rand[11000:140000]

recipes = rand[14000:21000]
recipes_train = rand[14000:18000]
recipes_test = rand[18000:21000]

gardening = rand[21000:28000]
gardening_train = rand[21000:25000]
gardening_test = rand[21000:28000]

# making the datasets

hik_vs_eth = np.vstack((hiking, ethereum))
print("Hiking vs ethereum matrix:", hik_vs_eth.shape)

hik_vs_rec = np.vstack((hiking, recipes))
print("Hiking vs recipes matrix:", hik_vs_rec.shape)

hik_vs_gar = np.vstack((hiking, gardening))
print("Hiking vs recipes matrix:", hik_vs_gar.shape)

eth_vs_rec = np.vstack((ethereum, recipes))
print("Ethereum vs recipes matrix:", eth_vs_rec.shape)

eth_vs_gar = np.vstack((ethereum, gardening))
print("Ethereum vs recipes matrix:", eth_vs_gar.shape)

rec_vs_gar = np.vstack((recipes, gardening))
print("Ethereum vs recipes matrix:", rec_vs_gar.shape, "\n")

# PCA random documents

he_scal=scaler.fit_transform(hik_vs_eth)
# print("hik_vs_eth scal:", he_scal.shape)

hr_scal=scaler.fit_transform(hik_vs_rec)
# print("hik_vs_rec scal:", hr_scal.shape)

hg_scal=scaler.fit_transform(hik_vs_gar)
# print("hik_vs_gar scal:", Z_scal.shape)

er_scal=scaler.fit_transform(eth_vs_rec)
# print("eth_vs_rec scal", er_scal.shape)

eg_scal=scaler.fit_transform(eth_vs_gar)
# print("eth_vs_gar scal", eg_scal.shape)

rg_scal=scaler.fit_transform(rec_vs_gar)
# print("rec_vs_gar scal", rg_scal.shape, "\n")

from sklearn.decomposition import PCA
pca = PCA(384) # 384 components

he_pca=pca.fit_transform(he_scal)

hr_pca=pca.fit_transform(hr_scal)

hg_pca=pca.fit_transform(hg_scal)

er_pca=pca.fit_transform(er_scal)

eg_pca=pca.fit_transform(eg_scal)

rg_pca=pca.fit_transform(rg_scal)


with open('pca_he.pkl', "wb") as fOut:
    pickle.dump(he_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_hr.pkl', "wb") as fOut:
    pickle.dump(hr_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_hg.pkl', "wb") as fOut:
    pickle.dump(hg_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_er.pkl', "wb") as fOut:
    pickle.dump(er_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_eg.pkl', "wb") as fOut:
    pickle.dump(eg_pca, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca_rg.pkl', "wb") as fOut:
    pickle.dump(eg_scal, fOut, protocol=pickle.HIGHEST_PROTOCOL)


he_pca = pickle.load(open("pca_he.pkl",'rb'))
print("he_pca:", he_pca.shape)

hr_pca = pickle.load(open("pca_hr.pkl",'rb'))
print("hr_pca:", hr_pca.shape)

hg_pca = pickle.load(open("pca_hg.pkl",'rb'))
print("hg_pca:", hg_pca.shape)

er_pca = pickle.load(open("pca_er.pkl",'rb'))
print("er_pca:", er_pca.shape)

eg_pca = pickle.load(open("pca_eg.pkl",'rb'))
print("eg_pca:", eg_pca.shape)

rg_pca = pickle.load(open("pca_rg.pkl",'rb'))
print("rg_pca:", rg_pca.shape)

# topical dataset, original matrices and PCA matrices

def sort_tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
         
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j]= tup[j + 1]
                tup[j + 1]= temp
    return tup

def evaluate_dimensions(x):
    scores = []
    for col in x.T: # trasposed matrix: docs in the rows, dims in the col
        mn = np.mean(col)
        c1h = len(np.where(col[:7000] > mn)[0]) # look at the records in the first cluster that are above the mean
        scores.append(abs(0.5 - c1h / 7000)) 
    #print('SCORES:',scores)
    ind = np.argpartition(scores, -5)[-5:]
    best = [(i,scores[i]) for i in ind]
    sort = sort_tuple(best)   
    print('BEST DIM:', sort[-1:])
    
    best_indices = [i[0] for i in sort] 
    # print("BEST indices:", best_indices)

# original matrices

evaluate_dimensions(hik_vs_eth)
evaluate_dimensions(hik_vs_rec)
evaluate_dimensions(hik_vs_gar)
evaluate_dimensions(eth_vs_rec)
evaluate_dimensions(eth_vs_gar)
evaluate_dimensions(rec_vs_gar)

# PCA matrices

evaluate_dimensions(he_pca)
evaluate_dimensions(hr_pca)
evaluate_dimensions(hg_pca)
evaluate_dimensions(er_pca)
evaluate_dimensions(eg_pca)
evaluate_dimensions(rg_pca)


