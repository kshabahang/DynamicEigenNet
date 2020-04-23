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
    N = V

    C1 = np.zeros((N, N))
    C2 = np.zeros((N, N))
    O = np.zeros((K*V, K*V))

    for i in range(len(sents)):
        sent = sents[i].split()
        s = np.zeros(V)
        o = np.zeros(K*V)

        for j in range(len(sent)):
            w = sent[j]
            o[j*V + I[w]] += 1
            s[I[w]] += 1
#        x /= np.linalg.norm(x)
        O += np.outer(o,o)
        C1 += np.outer(s,s)
        s2 = s.dot(C1)
        s2 /= np.linalg.norm(s2)
        C2 += (np.outer(s2,s2) - np.outer(s,s))

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(C1)
    ax2.imshow(C2)
    fig.show()

    vocab = np.array(vocab)

    s = np.zeros(V)
    o = np.zeros(K*V)
    q = "who is bellamira loved by".split()

    for i in range(len(q)):
        w = q[i]
        o[i*V + I[w]] += a[i]
        s[I[w]] += a[i]


    s2 = s.dot(C2)
    isort = np.argsort(s2)[::-1][:5]
    print(list(zip(vocab[isort], s2[isort])))









        




