import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

def load_original_embeddings():
    print("Loading original embeddings")
        
    abortion_dataset = pickle.load(open("./data/all_abortion.pkl",'rb'))
    abortion_dataset = abortion_dataset['embeddings']
    train_abortion_dataset = np.vstack((abortion_dataset[:7000], abortion_dataset[10000:17000]))
    test_abortion_dataset = np.vstack((abortion_dataset[7000:10000], abortion_dataset[17000:20000]))
    print("Train abortion_dataset:", train_abortion_dataset.shape, "\n", "Test abortion_dataset:", test_abortion_dataset.shape, "\n")

    climate_dataset = pickle.load(open("./data/all_climate.pkl",'rb'))
    climate_dataset = climate_dataset['embeddings']
    train_climate_dataset = np.vstack((climate_dataset[:7000], climate_dataset[10000:17000]))
    test_climate_dataset = np.vstack((climate_dataset[7000:10000], climate_dataset[17000:20000]))
    print("Train climate_dataset:", train_climate_dataset.shape, "\n", "Test climate_dataset:", test_climate_dataset.shape, "\n")

    vax_dataset = pickle.load(open("./data/all_vax.pkl",'rb'))
    vax_dataset = vax_dataset['embeddings']
    train_vax_dataset = np.vstack((vax_dataset[:7000], vax_dataset[10000:17000]))
    test_vax_dataset = np.vstack((vax_dataset[7000:10000], vax_dataset[17000:20000]))
    print("Train vax_dataset:", train_vax_dataset.shape, "\n", "Test vax_dataset:", test_vax_dataset.shape, "\n")

    politics_dataset = pickle.load(open("./data/all_politics.pkl",'rb'))
    politics_dataset = politics_dataset['embeddings']
    train_politics_dataset = np.vstack((politics_dataset[:7000], politics_dataset[10000:17000]))
    test_politics_dataset = np.vstack((politics_dataset[7000:10000], politics_dataset[17000:20000]))
    print("Train politics_dataset:", train_politics_dataset.shape, "\n", "Test politics_dataset:", test_politics_dataset.shape, "\n")

    return test_abortion_dataset, test_climate_dataset, test_vax_dataset, test_politics_dataset



def load_topical_embeddings():
    print("Loading topical embeddings")
    path = "./data/random_docs.pkl" 

    t1 = pickle.load(open(path, "rb"))
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


    print("he train shape:", he_train.shape)
    print("he test shape", he_test.shape)
    print("hr train shape", hr_train.shape)
    print("hr test shape", hr_test.shape)
    print("hg train shape", hg_train.shape)
    print("hg test shape", hg_test.shape)
    print("er train shape", er_train.shape)
    print("er test shape", er_test.shape)
    print("eg train shape", eg_train.shape)
    print("eg test shape", eg_test.shape)
    print("rg train shape", rg_train.shape)
    print("rg test shape", rg_test.shape)

    return he_train, he_test

def acquire_and_standardize_matrices():
    abortion_dataset = pickle.load(open("./data/all_abortion.pkl",'rb'))
    abortion_dataset = abortion_dataset['embeddings']
    print("Abortion original matrix abo:", abortion_dataset.shape)

    climate_dataset = pickle.load(open("./data/all_climate.pkl",'rb'))
    climate_dataset = climate_dataset['embeddings']
    print("Climate original matrix cli:", climate_dataset.shape)

    vax_dataset = pickle.load(open("./data/all_vax.pkl",'rb'))
    vax_dataset = vax_dataset['embeddings']
    print("Vaccine original matrix vax:", vax_dataset.shape)

    politics_dataset = pickle.load(open("./data/all_politics.pkl",'rb'))
    politics_dataset = politics_dataset['embeddings']
    print("Politics original matrix pol:", politics_dataset.shape, "\n")

    path = "./data/random_docs.pkl"

    t1 = pickle.load(open(path, "rb"))
    t1 = t1["embeddings"]

    hiking = t1[:7000]
    ethereum = t1[7000:14000]
    recipes = t1[14000:21000]
    gardening = t1[21000:28000]

# making the datasets
                     
    he_dataset = np.vstack((hiking, ethereum))                     
    print("Hiking vs ethereum matrix:", he_dataset.shape)
           
    hr_dataset = np.vstack((hiking, recipes))
    print("Hiking vs recipes matrix:", hr_dataset.shape)
                     
    hg_dataset = np.vstack((hiking, gardening))
    print("Hiking vs gardening matrix:", hg_dataset.shape)

    er_dataset = np.vstack((ethereum, recipes))
    print("Ethereum vs recipes matrix:", er_dataset.shape)
                     
    eg_dataset = np.vstack((ethereum, gardening))
    print("Ethereum vs gardening matrix:", eg_dataset.shape)

    rg_dataset = np.vstack((recipes, gardening))
    print("Recipes vs gardening matrix:", rg_dataset.shape, "\n")

#2 - standardize and save the original matrices standardized

#   from sklearn.preprocessing import StandardScaler
#   scaler=StandardScaler()

    abo_scal=scaler.fit_transform(abortion_dataset) # abortion
    cli_scal=scaler.fit_transform(climate_dataset) # climate
    vax_scal=scaler.fit_transform(vax_dataset) # vaccine
    pol_scal=scaler.fit_transform(politics_dataset) # politics

    he_scal=scaler.fit_transform(he_dataset)
    hr_scal=scaler.fit_transform(hr_dataset)
    hg_scal=scaler.fit_transform(hg_dataset)
    er_scal=scaler.fit_transform(er_dataset)
    eg_scal=scaler.fit_transform(eg_dataset)
    rg_scal=scaler.fit_transform(rg_dataset)

    return abortion_dataset, climate_dataset, vax_dataset, politics_dataset, he_dataset, hr_dataset, hg_dataset, er_dataset, eg_dataset, rg_dataset, abo_scal, cli_scal, vax_scal, pol_scal, he_scal, hr_scal, hg_scal, er_scal, eg_scal, rg_scal


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

def evaluate_dimensions(matrix):
    scores = []
    for col in matrix.T: # trasposed matrix: docs in the rows, dims in the col
        mn = np.mean(col)
        c1h = len(np.where(col[:10000] > mn)[0]) # look at the records in the first cluster that are above the mean
        scores.append(abs(0.5 - c1h / 10000)) 
    #print('SCORES:',scores)
    ind = np.argpartition(scores, -5)[-5:]
    best = [(i,scores[i]) for i in ind]
    sort = sort_tuple(best)   
    print('BEST DIM:', sort[-1:])
    
    best_indices = [i[0] for i in sort] 
    print("BEST indices:", best_indices)

# print("DIMS for abortion dataset:", evaluate_dimensions(abo_scal))
# print("DIMS for abortion dataset:", scaler.fit_transform(abortion_dataset))

evaluate_dimensions(cli_scal), evaluate_dimensions(vax_scal), evaluate_dimensions(pol_scal), evaluate_dimensions(he_scal), evaluate_dimensions(hr_scal), evaluate_dimensions(hg_scal), evaluate_dimensions(hr_scal), evaluate_dimensions(er_scal), evaluate_dimensions(eg_scal), evaluate_dimensions(rg_scal)


