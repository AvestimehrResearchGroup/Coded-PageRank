from mpi4py import MPI
import numpy as np
import sys
import random
import threading
import time
import networkx as nx
import copy
import itertools
import cPickle as pickle
import gmpy2
import math

#################create a subgraph and then partition and then create worker dictionaries#############

K=10
N=12600
p=0.3

Dict={}
for i in range(N):
	Dict[i]=[]
for i in range(N):
	for j in range(i+1,N):
		sample=np.random.binomial(1, p, 1) #(n,p,number of samples)
		if sample[0]==1:
			Dict[i].append(j)
			Dict[j].append(i)

# creating separate files

Gdist= [None] * K
for j in range(K):
	Gdist[j]={}
	Gdist[j][1]=[]
	Gdist[j][2]=[]

for j in range(K):
	for i in range(j,N,K):
		Gdist[j][1].append(i)
		Gdist[j][2].append(Dict[i])

for j in range(K):
	i=j+1
	with open("Dict_%s.txt"%i, "wb") as myFile:
		pickle.dump(Gdist[j], myFile)
	myFile.close()

# to keep the dictionary for further proccessing for Coded case
i=0
with open("Dict_%s.txt"%i, "wb") as myFile:
	pickle.dump(Dict, myFile)
myFile.close()	

