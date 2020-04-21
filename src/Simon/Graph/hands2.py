from scipy.sparse import csr_matrix
from scipy.optimize import differential_evolution
import numpy as np
from numpy.linalg import norm
import warnings
import sys
warnings.filterwarnings("ignore")

words = []

indexes = {}

def eig(M):
    vtp1 = np.ones((M.shape[0], 1), float)
    vt = np.zeros((M.shape[0], 1), float)
    iterations = 0
    while norm(vtp1 - vt) > 1.e-15 and iterations < 1000:
      vt = vtp1
      vtp1 = M.dot(vt)
      f = norm(vtp1)
      vtp1 = vtp1 / f
      iterations += 1
    return f, vtp1

def eigenvalue(self):
    val, vec = self.eig()
    return val

def c(word):
  if word not in indexes:
    indexes[word] = len(indexes)
    words.append(word)
  return indexes[word]

M = csr_matrix((100,100))

def setWeight(word1, word2, value, verbose = False):
  try:
    M[c(word1), c(word2)] = value
    M[c(word2), c(word1)] = value
    if verbose:
      print(word1 + " " + word2 + " {0:1.2f}".format(value))
  except:
    print (word1, word2)
    sys.exit()

def allConnections(tokens, value, verbose=False):
  if verbose:
    print ("allConnections {}".format(" ".join(tokens)))
  for i in range(len(tokens)-1):
    for j in range(i+1, len(tokens)):
      setWeight(tokens[i], tokens[j], value)

def oneToMany(word, tokens, value, verbose=False):
  if verbose:
    print ("oneToMany {} onto {}".format(word, " ".join(tokens)))
  for token in tokens:
    setWeight(word, token, value)

def setWeights(I = 20, P = 3, S = 20, Os = 3, O = 0, verbose = False):

  # Order weights

  if verbose:
    print ("Order Weights")
  allConnections(["1john", "2loves", "3mary"], O, verbose)
  allConnections(["1bob", "2loves", "3sue"], O, verbose)
  allConnections(["1lance", "2loves", "3jess"], O, verbose)
  allConnections(["1lysander", "2loves", "3bellamira"], O, verbose)

  allConnections(["1who", "2is", "3mary", "4loved", "5by", "6john"], O, verbose)
  allConnections(["1who", "2is", "3sue", "4loved", "5by", "6bob"], O, verbose)
  allConnections(["1who", "2is", "3jess", "4loved", "5by", "6lance"], O, verbose)
  if verbose:
    print ("\nSTP Order Weights")
  allConnections(["1who", "2is", "3bellamira", "4loved", "5by"], Os, verbose)

  # Paradigmatic weights

  if verbose:
    print ("\nParadigmatic Weights")
  allConnections(["6lysander", "6john", "6bob", "6lance"], P, verbose)
  allConnections(["3bellamira", "3mary", "3sue", "3jess"], P, verbose)

  # Item weights

  if verbose:
    print ("\nItem Weights")
  oneToMany("Wbellamira", ["1bellamira", "2bellamira", "3bellamira", "4bellamira", "5bellamira", "6bellamira"], I, verbose)
  oneToMany("Wlysander", ["1lysander", "2lysander", "3lysander", "4lysander", "5lysander", "6lysander"], I, verbose)

  oneToMany("Wmary", ["1mary", "2mary", "3mary", "4mary", "5mary", "6mary"], I, verbose)
  oneToMany("Wjohn", ["1john", "2john", "3john", "4john", "5john", "6john"], I, verbose)

  oneToMany("Wsue", ["1sue", "2sue", "3sue", "4sue", "5sue", "6sue"], I, verbose)
  oneToMany("Wbob", ["1bob", "2bob", "3bob", "4bob", "5bob", "6bob"], I, verbose)

  oneToMany("Wjess", ["1jess", "2jess", "3jess", "4jess", "5jess", "6jess"], I, verbose)
  oneToMany("Wlance", ["1lance", "2lance", "3lance", "4lance", "5lance", "6lance"], I, verbose)

  oneToMany("Wwho", ["1who", "2who", "3who", "4who", "5who", "6who"], I, verbose)
  oneToMany("Wis", ["1is", "2is", "3is", "4is", "5is", "6is"], I, verbose)
  oneToMany("Wloved", ["1loved", "2loved", "3loved", "4loved", "5loved", "6loved"], I, verbose)
  oneToMany("Wby", ["1by", "2by", "3by", "4by", "5by", "6by"], I, verbose)

  # Syntagmatic Weights

  if verbose:
    print ("\nSyntagmatic Weights")
  allConnections(["Wbellamira", "Wlysander"], S, verbose)
  allConnections(["Wmary", "Wjohn"], S, verbose)
  allConnections(["Wsue", "Wbob"], S, verbose)
  allConnections(["Wjess", "Wlance"], S, verbose)

 
def printVec(v):
  pairs = {"1":[], "2":[], "3":[], "4":[], "5":[], "6":[], "W":[]}
  for i in range(len(indexes)):
    if v[i] > 0.01:
      pairs[words[i][0]].append((v[i][0], words[i][1:]))
  s = ""
  for bank in pairs.keys():
    pairs[bank].sort(reverse=True)
    s += bank + ": " + " ".join("{0} {1:.2f} ".format(w, v) for v, w in pairs[bank]) + "\n"
  return s
  
def objective(params):
  global evalcount
  setWeights(params[0], params[1], params[2], params[3], O)
  val, vec = eig(M)
  result = min(vec[c("6lysander")][0] / vec[c("6john")][0], 
               vec[c("6lysander")][0] / vec[c("6bellamira")][0],
               vec[c("4loved")][0] / vec[c("4bellamira")][0],
               vec[c("4loved")][0] / vec[c("4lysander")][0])
  result = -result
  evalcount += 1
  if evalcount % 100 == 0:
    sys.stdout.write(".")
    sys.stdout.flush()
  return result

bounds = [(0, 60), (0,60), (0,60), (0,60)]

O = 1. # O is fixed to establish scale

print("Optimizing")
evalcount = 0
result = differential_evolution(objective, bounds, workers = -1)
print()
print()
print("Min constraint distance = {0:1.3f}. If this number is greater than one we have found a solution that works.".format(-result.fun))
print ("I = {0:.1f} P = {1:.1f} S = {2:.1f} Os = {3:.1f} O = {4:.1f}".format(result.x[0], result.x[1], result.x[2], result.x[3], O))

print()
print ("Creating weight matrix")
print()
setWeights(result.x[0], result.x[1], result.x[2], result.x[3], O, verbose=True)
print()
val, vec = eig(M)
print("Result")
print()
print(printVec(vec))
