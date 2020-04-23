import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    print()
    sents = ["lysander loves bellamira", "john loves mary", "bob loves sue", "who is mary loved by john", "who is sue loved by bob"]
    K = 6
    T = 7
    vocab = list(set(" ".join(sents).split()))
    V = len(vocab)
    I = {vocab[i]:i for i in range(V)}
    N = (K+1)*V

    C1 = np.zeros((N, N))
    C2 = np.zeros((N, N))

    for i in range(len(sents)):
        sent = sents[i].split()
        x = np.zeros(N)
        for j in range(len(sent)):
            w = sent[j]
            x[j*V + I[w]] += 1
            x[K*V + I[w]] += 1
        C1 += np.outer(x,x)
        x2 = x.dot(C1)
        x2 /= np.linalg.norm(x2)
        C2 += (np.outer(x2,x2) - np.outer(x,x))

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(C1)
    ax2.imshow(C2)
    fig.show()

    vocab = np.array(vocab)

    x = np.zeros(N)
    x[I['lysander']] = 1
    x2 = x.dot(C2)
    for i in range(K+1):
        x2_i = x2[i*V:(i+1)*V]
        isort = np.argsort(x2_i)[::-1]
        res = list(zip(vocab[isort], x2_i[isort]))
        s = ''
        for j in range(5):
            s += "%15s %1.4f " % (res[j][0], res[j][1])
        print(s)

        




