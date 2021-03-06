import sys, os
from AssociativeNet import *
#from matplotlib import pyplot as plt
#plt.ion()
from progressbar import ProgressBar

import pickle
from copy import deepcopy
from collections import Counter
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

def save_csr(matrix, filename):
    matrix.data.tofile(filename + "_"+ str(type(matrix.data[0])).split(".")[-1].strip("\'>") + ".data")
    matrix.indptr.tofile(filename + "_"+ str(type(matrix.indptr[0])).split(".")[-1].strip("\'>") + ".indptr")
    matrix.indices.tofile(filename + "_" +str(type(matrix.indices[0])).split(".")[-1].strip("\'>") + ".indices")

def load_csr(fpath, shape, dtype=np.float64, itype=np.int32):
    data = np.fromfile(fpath + "_{}.data".format(str(dtype).split(".")[-1].strip("\'>")), dtype)
    indptr = np.fromfile(fpath + "_{}.indptr".format(str(itype).split(".")[-1].strip("\'>")), itype)
    indices = np.fromfile(fpath + "_{}.indices".format(str(itype).split(".")[-1].strip("\'>")), itype)

    return csr_matrix((data, indices, indptr), shape = shape)



if __name__ == "__main__":
    f = open("../rsc/bundles.pkl","rb")
    bundles = pickle.load(f)
    f.close()

    vocab = []
    wf = {}
    sents = []
    for i in range(len(bundles)):
        sent = ' '.join(bundles[i][0]).lower().split()
        for j in range(len(sent)):
            w = sent[j]
            if w not in vocab:
                vocab.append(w)
                wf[w] = 1
            else:
                wf[w] += 1
   
    vocab_trunc = []
    for i in range(len(vocab)):
        if wf[vocab[i]] > 1:
            vocab_trunc.append(vocab[i])
    vocab = vocab_trunc
    V = len(vocab)
    I = {vocab[i]:i for i in range(V)}



    N =V

    K = 2
    
    hparams = {"bank_labels":["t-{}".format(i) for i in range(K)],
               "eps":0.001, 
               "eta":1,
               "alpha":1,
               "beta":1,
               "V0":N,
               "N":N,
               "idx_predict":1,
               "numBanks":K, 
               "numSlots":K,
               "C":1,
               "mode":"numpy",
               "feedback":"stp",
               "gpu":False,
               "localist":True,
               "distributed":False,
               "explicit":False}
#    ANet = AssociativeNet(hparams)
#    
#    f = open("../rsc/TASA/vocab.txt", "r")
#    vocab = f.readlines()
#    ANet.vocab = [vocab[i].strip() for i in range(len(vocab))]
#    f.close()
#    ANet.I = {ANet.vocab[i]:i for i in range(len(vocab))}
#    
#
#    thetas = [0.25, 0.5, 1]
#    scores= []
#    for theta in thetas:
#        NIter = 10000
#        max_eig = []
#        pbar = ProgressBar(maxval = NIter).start()
#        for i in range(NIter):
#            x = np.zeros(N*K)
#            for k in range(K):
#                w = np.random.randint(N)
#                x[w] += 1
#            x = x/np.linalg.norm(x)
#            ANet.W[:, :] += np.outer(x,x)*(theta  -  ANet.W)
#            ei, ev = np.linalg.eig(ANet.W)
#            eig_max = max(abs(ei))
#            max_eig.append(eig_max)
#            pbar.update(i+1)
##        print(eig_max)
#        scores.append(max_eig)
#
#    fig = plt.figure()
#    for i in range(len(thetas)):
#        plt.plot(scores[i], label = thetas[i])
#    plt.legend()
#    fig.show()

        
        

