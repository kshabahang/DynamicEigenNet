from scipy.sparse import csr_matrix
from scipy.optimize import differential_evolution
import numpy as np
from numpy.linalg import norm
import warnings
import sys
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
import networkx as nx

words = []

indexes = {}

def eig(M, verbose=False):
    vtp1 = np.ones((M.shape[0], 1), float)
    vt = np.zeros((M.shape[0], 1), float)
    iterations = 0
    while norm(vtp1 - vt) > 1.e-10 and iterations < 1000:
      vt = vtp1
      vtp1 = M.dot(vt)
      f = norm(vtp1)
      vtp1 = vtp1 / f
      if verbose:
        sys.stdout.write("{0:1.5f} ".format(vtp1[c("6lysander"), 0]))
      iterations += 1
    sys.stdout.write (str(iterations) + " ")
    if iterations == 1000:
      #print ("Warning: eig did not converge. norm(vtp1-vt) = {} ".format(norm(vtp1 - vt)))
      return (None, None)
    else:
      return f, vtp1

def eigenvalue(self):
    val, vec = self.eig()
    return val

def c(word):
  if word not in indexes:
    indexes[word] = len(indexes)
    words.append(word)

    global color, color_map, nodes, node_lbls, G
    nodes.append(word)
#    color_map.append(color)
    node_lbls[len(indexes)] = word
    G.add_node(len(indexes), name=word)

  return indexes[word]

M = csr_matrix((100,100))


global color
global color_map
global G 
global nodes 
global node_lbls 

color = None
color_map = []
G = nx.Graph()
nodes = []
node_lbls = {}

def isSymmetric(M):
  symmetric = True
  for i in range(M.shape[0]-1):
    for j in range(i+1, M.shape[0]):
      symmetric = symmetric and M[i,j] == M[j,i]
  return symmetric

def setWeight(word1, word2, value, verbose = False):
  global color, color_map
  try:
    M[c(word1), c(word2)] = value
    M[c(word2), c(word1)] = value

    G.add_edge(c(word1), c(word2), w=value)
    color_map.append(color)
    if verbose:
      print(word1 + " " + word2 + " {0:1.2f}".format(value))
  except Exception as e:
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

def setWeights(I = 20, P = 3, S = 20, Os = 3, N = -1, O = 0, verbose = False):

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

  # Inhibition between banks

  if verbose:
    print ("\nInhibition Weights")
  allConnections(["1bellamira", "2bellamira", "3bellamira", "4bellamira", "5bellamira", "6bellamira"], N, verbose)
  allConnections(["1lysander", "2lysander", "3lysander", "4lysander", "5lysander", "6lysander"], N, verbose)

  allConnections(["1mary", "2mary", "3mary", "4mary", "5mary", "6mary"], N, verbose)
  allConnections(["1john", "2john", "3john", "4john", "5john", "6john"], N, verbose)

  allConnections(["1sue", "2sue", "3sue", "4sue", "5sue", "6sue"], N, verbose)
  allConnections(["1bob", "2bob", "3bob", "4bob", "5bob", "6bob"], N, verbose)

  allConnections(["1jess", "2jess", "3jess", "4jess", "5jess", "6jess"], N, verbose)
  allConnections(["1lance", "2lance", "3lance", "4lance", "5lance", "6lance"], N, verbose)

  allConnections(["1who", "2who", "3who", "4who", "5who", "6who"], N, verbose)
  allConnections(["1is", "2is", "3is", "4is", "5is", "6is"], N, verbose)
  allConnections(["1loved", "2loved", "3loved", "4loved", "5loved", "6loved"], N, verbose)
  allConnections(["1by", "2by", "3by", "4by", "5by", "6by"], N, verbose)
  
 
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
  setWeights(params[0], params[1], params[2], params[3], params[4], O)
  val, vec = eig(M)
  if val:
    result = min(vec[c("6lysander")][0] / vec[c("6john")][0], 
               vec[c("6lysander")][0] / vec[c("6bellamira")][0],
               vec[c("6lysander")][0] / vec[c("6who")][0],
               vec[c("4loved")][0] / vec[c("4bellamira")][0],
               vec[c("4loved")][0] / vec[c("4lysander")][0])
  else:
    result = -1000000.0
  result = -result
  evalcount += 1
  if evalcount % 100 == 0:
    sys.stdout.write(".")
    sys.stdout.flush()
  return result


def unitsInput(unit, vec):
  total = 0.0
  for i in range(len(words)):
    if abs(vec[i][0]*M[i,c(unit)]) > 0.000001:
      print ("{0} -> {1} {2:1.3f} x {3:1.3f} = {4:1.3f}".format(words[i], unit, vec[i][0], M[i, c(unit)], vec[i][0]*M[i, c(unit)]))
      total += vec[i][0]*M[i, c(unit)]
  print ("Total = {0:1.3f}".format(total))
  return total

bounds = [(0, 60), (0,60), (0,60), (0,60), (-60, 0)]

O = 1. # O is fixed to establish scale

print("Optimizing")
evalcount = 0
result = differential_evolution(objective, bounds, workers = -1)
print()
print()
print("Min constraint distance = {0:1.3f}. If this number is greater than one we have found a solution that works.".format(-result.fun))
print ("I = {0:.1f} P = {1:.1f} S = {2:.1f} Os = {3:.1f} N = {4:.1f} I = {5:.1f}".format(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], O))

print()
print ("Creating weight matrix")
print()
setWeights(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], O, verbose=True)
#setWeights(2.84976839,  8.29262612, 59.87033616, 16.97117621, -1.18271248, O, verbose=True)
print()
if not isSymmetric(M):
  print ("Warning: M is not symmetric.")
val, vec = eig(M, verbose=True)
print("Result")
print()
print(printVec(vec))

#pos=nx.spring_layout(G)





pos = {0:[0,0]}
node_cmap = ["white"]
shifts = {}
for i in range(len(words)):
    if words[i][0] in '1 2 3 4 5 6'.split():
        node_cmap.append('orange')
        #slots
        slot = int(words[i][0])

        if str(slot) + "slot" in shifts:
            shifts[str(slot) + "slot"] += 1
        else:
            shifts[str(slot) + "slot"] = 0

        pos[i+1] = [slot, shifts[str(slot) + "slot"]]

    elif words[i][0] == 'W':
        node_cmap.append('blue')
        #syntagmatic
        if "W" in shifts:
            shifts["W"] += 1
        else:
            shifts["W"] = 0
        pos[i+1] = [8, shifts["W"]]
    else:
        node_cmap.append('white')
        pos[i+1] = [0, 10]

nx.draw(G, pos, label=node_lbls, node_color=node_cmap)

nx.draw_networkx_labels(G,pos,node_lbls, font_size=16)
plt.show()




o1 = ["1john", "2loves", "3mary"] #O
o2 = ["1bob", "2loves", "3sue"] #O
o3 = ["1lance", "2loves", "3jess"] #O
o4 = ["1lysander", "2loves", "3bellamira"] #O


o5 = ["1who", "2is", "3mary", "4loved", "5by", "6john"] #O
o6 = ["1who", "2is", "3sue", "4loved", "5by", "6bob"] #O
o7 = ["1who", "2is", "3jess", "4loved", "5by", "6lance"] #O

O_assocs = [o1,o2,o3,o4,o5,o6,o7]

os1 = ["1who", "2is", "3bellamira", "4loved", "5by"] #Os

O_STP_assocs = [os1]

# Paradigmatic weights



p1 = ["6lysander", "6john", "6bob", "6lance"] #P
p2 = ["3bellamira", "3mary", "3sue", "3jess"] #P

P_assocs = [p1, p2]

# Item weights

i1 = ("Wbellamira", ["1bellamira", "2bellamira", "3bellamira", "4bellamira", "5bellamira", "6bellamira"]) #I
i2 = ("Wlysander", ["1lysander", "2lysander", "3lysander", "4lysander", "5lysander", "6lysander"]) #I

i3 = ("Wmary", ["1mary", "2mary", "3mary", "4mary", "5mary", "6mary"]) #I
i4 = ("Wjohn", ["1john", "2john", "3john", "4john", "5john", "6john"]) #I

i5 = ("Wsue", ["1sue", "2sue", "3sue", "4sue", "5sue", "6sue"]) #I
i6 = ("Wbob", ["1bob", "2bob", "3bob", "4bob", "5bob", "6bob"]) #I

i7 = ("Wjess", ["1jess", "2jess", "3jess", "4jess", "5jess", "6jess"]) #I
i8 = ("Wlance", ["1lance", "2lance", "3lance", "4lance", "5lance", "6lance"]) #I

i9 = ("Wwho", ["1who", "2who", "3who", "4who", "5who", "6who"]) #I
i10 = ("Wis", ["1is", "2is", "3is", "4is", "5is", "6is"]) #I
i11 = ("Wloved", ["1loved", "2loved", "3loved", "4loved", "5loved", "6loved"]) #I
i12 = ("Wby", ["1by", "2by", "3by", "4by", "5by", "6by"]) #I

I_assocs = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12]

# Syntagmatic Weights



s1 = ["Wbellamira", "Wlysander"] #S
s2 = ["Wmary", "Wjohn"] #S
s3 = ["Wsue", "Wbob"] #S
s4 = ["Wjess", "Wlance"] #S 


S_assocs = [s1,s2,s3,s4]
# Inhibition between banks



n1 = ["1bellamira", "2bellamira", "3bellamira", "4bellamira", "5bellamira", "6bellamira"] #N
n2 = ["1lysander", "2lysander", "3lysander", "4lysander", "5lysander", "6lysander"] #N

n3 = ["1mary", "2mary", "3mary", "4mary", "5mary", "6mary"] #N
n4 = ["1john", "2john", "3john", "4john", "5john", "6john"] #N

n5 = ["1sue", "2sue", "3sue", "4sue", "5sue", "6sue"] #N
n6 = ["1bob", "2bob", "3bob", "4bob", "5bob", "6bob"] #N

n7 = ["1jess", "2jess", "3jess", "4jess", "5jess", "6jess"] #N
n8 = ["1lance", "2lance", "3lance", "4lance", "5lance", "6lance"] #N

n9 = ["1who", "2who", "3who", "4who", "5who", "6who"] #N
n10 = ["1is", "2is", "3is", "4is", "5is", "6is"] #N
n11 = ["1loved", "2loved", "3loved", "4loved", "5loved", "6loved"] #N
n12 = ["1by", "2by", "3by", "4by", "5by", "6by"] #N

N_assoc = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12]
