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
from array import array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

K=10
N=12600
prank0=1.0/N;
mach=[]
for j in range(K):
	mach.append(j)

if rank == 0:
	masterComm = comm.Split( 0, rank );
	tmap=0.0
	tshuffle=0.0
	treduce=0.0
	print rank
	iterflag=1
	itercnt=0
	##########################################
	comm.Barrier()
	#################### MAP PHASE ####################
	recvmsg1 = comm.gather(0, root=0)
	tmap=max(recvmsg1[1:])
	print 'tmap',tmap
	
	##################### Packing #####################
	recvmsg1 = comm.gather(0, root=0)
	tpacking=max(recvmsg1[1:])
	print 'tpacking',tpacking
	
	#################### Shuffle Phase ################

	recvmsg1 = comm.gather(0, root=0)
	tshuffle=max(recvmsg1[1:])
	print 'tshuffle',tshuffle
	
	################## Unpack Phase ##################
	
	recvmsg1 = comm.gather(0, root=0)
	tunpack=max(recvmsg1[1:])
	print 'tunpack',tunpack


	#################### Reduce Phase #################
	
	recvmsg1 = comm.gather(0, root=0)
	treduce=max(recvmsg1[1:])
	print 'treduce',treduce	
		
else:
	workerComm = comm.Split( 1, rank );
	################################################
	pval=prank0
	# reducer list
	i=rank
	###############Loading the subgraph dictionary into memory##############
	with open("Dict_%s.txt"%i, "rb") as myFile:
		G = pickle.load(myFile)
	myFile.close()
	listKeys=G[1]
	listNeighboursList=G[2]
	listNeighboursSize=[] # to store the size of neighbour list
	for i in listNeighboursList:
		listNeighboursSize.append(len(i))
	inputPartitionCollection=[None] * (K)
	destdict=[None] * (K)
	srcdict1=[None] * (K)
	srcdict=[None] * (K)
	for j in range(K):
		destdict[j]={}
		srcdict1[j]={}
		srcdict[j]={}
		inputPartitionCollection[j]={}
	iterrankc={}
	
	################### build dictionary inputPartitionCollection #############################
	
	for k in range(K):
		inputPartitionCollection[k][1]=[]
		inputPartitionCollection[k][2]=[]
		
	i1=rank-1
	for i2 in range(i1,N,K):
		iterrankc[i2]=prank0
	comm.Barrier()
	
	############################  Map phase ################################ 
	t0=time.time()
	for i1,i2,i3 in itertools.izip(listKeys,listNeighboursList,listNeighboursSize):
		pvalue=iterrankc[i1]
		new=pvalue/i3
		for i4 in i2:
			i5=i4%K
			inputPartitionCollection[i5][1].append(i4)
			inputPartitionCollection[i5][2].append(new)
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)
	
	############################Packing Phase##############################
	t0=time.time()
	for j in range(K):
		destdict[j][1]=np.array(inputPartitionCollection[j][1],dtype='i').dumps()
		destdict[j][2]=np.array(inputPartitionCollection[j][2],dtype='d').dumps()
		destdict[j][3]=[len(destdict[j][1]),len(destdict[j][2])]
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)

	############################ Shuffle Phase #############################
	t0=time.time()
	for j in range(K):
		#Send the data to other machines
		workerComm.Barrier()
		if ((j+1)==rank):
			for c1 in range(K):
				if (c1+1)==rank:
					continue
				workerComm.send(destdict[c1][3], dest=c1)
				workerComm.Send([destdict[c1][1],MPI.CHAR], dest=c1)
				workerComm.Send([destdict[c1][2],MPI.CHAR], dest=c1)
		#Receive data from other machines
		else:
			#rwSHS=comm.irecv(source=j+1,tag=j)
			#srcdict[j]=rwSHS.wait()
			temp = workerComm.recv(source=j)
			#temp3=temp[1] # buffer large enough for double and so for int
			buf = array('c', '\0') * temp[1]
			r = workerComm.Irecv([buf,MPI.CHAR],source=j)
			r.Wait()
			srcdict1[j][1]=buf[:temp[0]].tostring()
			
			r = workerComm.Irecv([buf,MPI.CHAR],source=j)
			r.Wait()
			srcdict1[j][2]=buf[:temp[1]].tostring()
			
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)

	############################ Unpack Phase ##############################
	t0=time.time()
	for i1 in range(K):
		if (i1+1)==rank:
			continue
		else:
			srcdict[i1][1]=np.loads(srcdict1[i1][1])
			srcdict[i1][2]=np.loads(srcdict1[i1][2])

	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)
	
	############################ Reduce Phase ##############################
	for k1 in iterrankc:
		iterrankc[k1]=0.0
	t0=time.time()
	for c1 in range(K):
		if ((c1+1)==rank):
			ltemp1=inputPartitionCollection[c1][1]
			ltemp2=inputPartitionCollection[c1][2]
			#print 'inside'
			for c3,c4 in itertools.izip(ltemp1,ltemp2):
			
				iterrankc[c3]+=c4
		# go over other received intermediate values	
		else:
			ltemp3=srcdict[c1][1].tolist()
			ltemp4=srcdict[c1][2].tolist()
			for c3,c4 in itertools.izip(ltemp3,ltemp4):
				iterrankc[c3]+=c4
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)
