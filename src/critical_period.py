import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from progressbar import ProgressBar
import os

if __name__ == "__main__":


    corpus = "../rsc/TASA/TASA.txt"
    vocab = []
    I = {}
    wf = {}
    V = 0
    O = 3
    N=20000
    C = lil_matrix((N*O, N*O))
    l = 0


    os.system("wc " + corpus + " > line_counts.dat" )

    f = open("line_counts.dat", "r")
    line_counts = int(f.readlines()[0].split()[0])
    f.close()
    pbar = ProgressBar(maxval=line_counts).start()


    with open(corpus,"r") as f:
        for line in f:
            ws = line.split()
            for k in range(int(len(ws)/O)):
                window = ws[k*O:(k+1)*O]
                for i in range(O):
                    if window[i] not in I:
                        vocab.append(window[i])
                        I[window[i]] = V
                        V += 1
                    for j in range(O):
                        if window[j] not in I:
                            vocab.append(window[j])
                            I[window[j]] = V
                            V += 1
                        C[N*i + I[window[i]],N*j+ I[window[j]]] += 1
            l+=1
            pbar.update(l)


