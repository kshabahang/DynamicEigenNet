from scipy.sparse import csr_matrix
from scipy.optimize import differential_evolution
import numpy as np
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

words = ["who1", "lysander1", "bellamira1",
         "is2", "lysander2", "bellamira2",
         "bellamira3", "lysander3", "mary3", "sue3", "jess3",
         "loved4", "bellamira4", "lysander4",
         "by5", "bellamira5", "lysander5",
         "lysander6", "bellamira6", "john6", "bob6", "lance6",
         "lysanderW", "bellamiraW"]

indexes = dict((w, i) for i, w in enumerate(words)) 

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

M = csr_matrix((30,30))

def setWeight(word1, word2, value):
  M[c(word1), c(word2)] = value
  M[c(word2), c(word1)] = value

def setWeights(I = 20, P = 3, S = 20, Os = 3, O = 1):

  # Order weights

  setWeight("who1", "is2", Os)
  setWeight("who1", "bellamira3", Os)
  setWeight("who1", "mary3", O)
  setWeight("who1", "sue3", O)
  setWeight("who1", "jess3", O)
  setWeight("who1", "loved4", Os)
  setWeight("who1", "by5", Os)
  setWeight("who1", "john6", O)
  setWeight("who1", "bob6", O)
  setWeight("who1", "lance6", O)
  setWeight("is2", "bellamira3", Os)
  setWeight("is2", "mary3", O)
  setWeight("is2", "sue3", O)
  setWeight("is2", "jess3", O)
  setWeight("is2", "loved4", Os)
  setWeight("is2", "by5", Os)
  setWeight("is2", "john6", O)
  setWeight("is2", "bob6", O)
  setWeight("is2", "lance6", O)
  setWeight("bellamira3", "loved4", Os)
  setWeight("bellamira3", "by5", Os)
  setWeight("bellamira3", "john6", O)
  setWeight("bellamira3", "bob6", O)
  setWeight("bellamira3", "lance6", O)
  setWeight("mary3", "loved4", O)
  setWeight("mary3", "by5", O)
  setWeight("mary3", "john6", O)
  setWeight("mary3", "bob6", O)
  setWeight("mary3", "lance6", O)
  setWeight("sue3", "loved4", O)
  setWeight("sue3", "by5", O)
  setWeight("sue3", "john6", O)
  setWeight("sue3", "bob6", O)
  setWeight("sue3", "lance6", O)
  setWeight("jess3", "loved4", O)
  setWeight("jess3", "by5", O)
  setWeight("jess3", "john6", O)
  setWeight("jess3", "bob6", O)
  setWeight("jess3", "lance6", O)
  setWeight("loved4", "by5", Os)
  setWeight("loved4", "john6", O)
  setWeight("loved4", "bob6", O)
  setWeight("loved4", "lance6", O)
  setWeight("by5", "john6", O)
  setWeight("by5", "bob6", O)
  setWeight("by5", "lance6", O)

  # Paradigmatic weights

  setWeight("john6", "bob6", P)
  setWeight("john6", "lance6", P)
  setWeight("john6", "lysander6", P)
  setWeight("bob6", "lance6", P)
  setWeight("bob6", "lysander6", P)
  setWeight("lance6", "lysander6", P)

  setWeight("bellamira3", "mary3", P)
  setWeight("bellamira3", "sue3", P)
  setWeight("bellamira3", "jess3", P)
  setWeight("mary3", "sue3", P)
  setWeight("mary3", "jess3", P)
  setWeight("sue3", "jess3", P)

  # Item weights

  setWeight("bellamira1", "bellamiraW", I)
  setWeight("bellamira2", "bellamiraW", I)
  setWeight("bellamira3", "bellamiraW", I)
  setWeight("bellamira4", "bellamiraW", I)
  setWeight("bellamira5", "bellamiraW", I)
  setWeight("bellamira6", "bellamiraW", I)
  setWeight("lysander1", "lysanderW", I)
  setWeight("lysander2", "lysanderW", I)
  setWeight("lysander3", "lysanderW", I)
  setWeight("lysander4", "lysanderW", I)
  setWeight("lysander5", "lysanderW", I)
  setWeight("lysander6", "lysanderW", I)

  # Syntagmatic Weights

  setWeight("bellamiraS", "lysanderS", S)
 
def printVec(v):
  s = ""
  for i in range(len(indexes)):
    if v[i] > 0.01:
      s += "{0} {1:.2f}  ".format(words[i], v[i][0])
  return s
  
def objective(params):
  setWeights(params[0], params[1], params[2], params[3], 1.)
  val, vec = eig(M)
  result = min(vec[c("lysander6")][0] - vec[c("john6")][0], 
               vec[c("loved4")][0] - vec[c("bellamira4")][0],
               vec[c("loved4")][0] - vec[c("lysander4")][0])
  result = -result
  return result

bounds = [(0, 40), (0,40), (0,40), (0,40)]

print("Optimizing ..")
result = differential_evolution(objective, bounds)

print("Min constraint distance = {0:1.3f}. If this number is positive we have found a solution that works.".format(-result.fun))
print ("I = {0:.1f} P = {1:.1f} S = {2:.1f} Os = {3:.1f} O = {4:.1f}".format(result.x[0], result.x[1], result.x[2], result.x[3], 1.))

setWeights(result.x[0], result.x[1], result.x[2], result.x[3], 1.)
val, vec = eig(M)
print(printVec(vec))
