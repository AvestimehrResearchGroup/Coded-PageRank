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

# To create the dictionaries for the workers in the coded case
# N is a multiple of K-choose-r

with open("Dict_0.txt", "rb") as myFile:
	Gdict = pickle.load(myFile)
K=10
N=12600
r=3 

kcr=int(gmpy2.comb(K,r))
g=N/kcr
fid=[x for x in range(1,kcr+1)]
ltemp1=[x for x in range(1,K+1)]
ltemp2=itertools.combinations(ltemp1, r)
WSubsetR={}
fidWSet={}
fidNodes={}
Gdist= [None] * K
for j in range(K):
	Gdist[j]={}

i1=1
for x in ltemp2:
	WSubsetR[i1]=x
	fidWSet[x]=i1
	i1+=1

# fidNodes
i2=0
for i1 in range(1,kcr+1):
	fidNodes[i1]=tuple([x for x in range(i2*g,(i2+1)*g)])
	i2+=1

for i1 in fid: 
	listKeys=[]
	listNeighboursList=[]
	listNeighboursSize=[]

	ltemp1=fidNodes[i1]

	for i2 in ltemp1:
		listKeys.append(i2)
		listNeighboursList.append(Gdict[i2])
		listNeighboursSize.append(len(Gdict[i2]))

	ltemp2=WSubsetR[i1]
	for i2 in ltemp2:
		Gdist[i2-1][i1]={}
		Gdist[i2-1][i1][1]=listKeys
		Gdist[i2-1][i1][2]=listNeighboursList
		Gdist[i2-1][i1][3]=listNeighboursSize

for j in range(K):
	print j
	i=j+1
	with open("DictC4_%s.txt"%i, "wb") as myFile:
		pickle.dump(Gdist[j], myFile)
	myFile.close()
	

