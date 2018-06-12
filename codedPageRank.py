from mpi4py import MPI
import numpy as np
import sys
import random
import threading
import time
import networkx as nx
import copy
import gmpy2
import math
import itertools
import cPickle as pickle
from array import array
import gc
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

r=3 # Computation load
K=10 # number of workers
N=12600 # number of graph nodes

# N is a multiple of K-choose-r
s=r+1

Prank0=1.0/N;
if rank==0:
	masterComm = comm.Split( 0, rank );
	comm.Barrier()
	
	recvmsg1 = comm.gather(0, root=0)
	print recvmsg1
	timeMap=max(recvmsg1[1:])
	print 'map',timeMap

	recvmsg1 = comm.gather(0, root=0)
	timeEncoding=max(recvmsg1[1:])
	print 'encode and pack',timeEncoding

	recvmsg1 = comm.gather(0, root=0)
	timeShuffle=max(recvmsg1[1:])
	print 'shuffle',timeShuffle

	recvmsg1 = comm.gather(0, root=0)
	timeUnpack=max(recvmsg1[1:])
	print 'unpack',timeUnpack
	
	recvmsg1 = comm.gather(0, root=0)
	timeDecoding=max(recvmsg1[1:])
	print 'decode',timeDecoding

	recvmsg1 = comm.gather(0, root=0)
	timeReduce=max(recvmsg1[1:])# +sum(recvmsg1[1:])+max(recvmsg1[1:])
	print 'reduce',timeReduce

else:
	################################## Creating workerComm ##################################
	workerComm = comm.Split( 1, rank );
	
	###################################### Variables ########################################
	# nodes indexed from 0 to N-1
	fid=[]  
	WSubsetR={} 
	WSubsetS=[] 
	fidWSet={} 
	fidNodes={}  
	inputPartitionCollection={} 
	encodePreData={} 
	encodeDataSend={} 
	enDataRecv={} 
	enDataRecv1={} 
	PrankMap={} 
	PrankReduce={} 
	LocalList={} 
	keyDict={}
	valueDict={}
	ltemp1=[] 
	ltemp2=[] 
	ltemp3=[]  
	ltemp4=[] 
	destPrank=[None]*(K) 
	srcPrank=[None]*(K) 
	multicastGroupMap={} 
	i1=0 
	i2=0 
	i3=0 
	i4=0 
	i5=0 
	i6=0 
	maxnumber=None 
	kcr=int(gmpy2.comb(K,r))
	g=N/kcr
	
	#################################### Loading sub-graph dictionary into memory ###################################
	
	with open("DictC3_%s.txt"%rank, "rb") as myFile: # change 3 to suitable r as required
		GdictLocal = pickle.load(myFile)
	myFile.close()

	################################ Initializing Data ##############################
	Gdist= [None] * K
	fid=[x for x in range(1,kcr+1)]
	
	# WSubsetR={}
	ltemp1=[x for x in range(1,K+1)]
	ltemp2=itertools.combinations(ltemp1, r)
	i1=1
	for x in ltemp2:
		WSubsetR[i1]=set(x) # used in reduce stage
		fidWSet[x]=i1
		i1+=1

	# WSubsetS
	kcs=int(gmpy2.comb(K,s))
	ltemp1=[x for x in range(1,K+1)]
	ltemp2=itertools.combinations(ltemp1, s)
	
	for x in ltemp2:
		WSubsetS.append(list(x))


	# multicastGroupMap
	for i1 in WSubsetS:
		ltemp1=i1
		if rank in ltemp1:
			color=1
		else:
			color=0
		multicastGroupMap[tuple(i1)]=workerComm.Split( color, rank );

	# destPrank
	for i1 in range(K):
		destPrank[i1]={}

	# srcPrank
	for i1 in range(K):
		srcPrank[i1]={}	

	# fidNodes
	i2=0
	for i1 in range(1,kcr+1):
		fidNodes[i1]=tuple([x for x in range(i2*g,(i2+1)*g)])
		i2+=1

	# inputPartitionCollection
	for i1 in GdictLocal:		
		for i2 in range(1,K+1):
			inputPartitionCollection[i1,i2]={}
	
	# PrankMap
	for i1 in GdictLocal:
		listKeys=GdictLocal[i1][1]
		for i2 in listKeys:
			PrankMap[i2]=Prank0
	
	# PrankReduce
	i1=rank-1
	for i2 in range(i1,N,K):
		PrankReduce[i2]=0.0

	# LocalList
	LocalList[1]=[]
	LocalList[2]=[]
	
	# enDataRecv, enDataRecv1
	i1=0
	for i2 in WSubsetS:
		i1+=1
		if rank in i2:
			for i3 in i2:
				if rank==i3:
					continue
				else:
					enDataRecv[i1,i3]={}
					enDataRecv1[i1,i3]={}
		else:
			continue
	comm.Barrier()
	
		######################################## Map Phase #########################################
	
	t0=time.time()
	for i1 in GdictLocal:
		fid1=i1
		for i2 in range(K):
			Gdist[i2]={}
			Gdist[i2][1]=[]
			Gdist[i2][2]=[]
		listKeys=GdictLocal[i1][1]
		listNeighboursList=GdictLocal[i1][2]
		listNeighboursSize=GdictLocal[i1][3]
		for i2,i3,i4 in itertools.izip(listKeys,listNeighboursList,listNeighboursSize):
			pvalue=PrankMap[i2]
			new=pvalue/i4
			for i5 in i3:
				i6=i5%K
				Gdist[i6][1].append(i5)
				Gdist[i6][2].append(new) 
		for i2 in range(K):
			inputPartitionCollection[fid1,(i2+1)][1]=Gdist[i2][1]
			inputPartitionCollection[fid1,(i2+1)][2]=Gdist[i2][2]
	
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)

	############################### Clear memory ###########################
	
	for i1 in GdictLocal:	
		fnodes=WSubsetR[i1]
		for i2 in fnodes:
			if i2==rank:
				continue
			else:
				inputPartitionCollection[i1,i2]=None

	# GdictLocal.clear() use this if the memory is small
	
	################################### Encoding and Packing for Shuffle###################################
	workerComm.Barrier()
	t0=time.time()
	i1=0
	for i11 in WSubsetS: # going over each s-sized subset
		i1+=1
		ltemp1=i11
		
		if rank in ltemp1:
			maxsize=0
			for i2 in ltemp1: # going over the nodes in s-sized subset , destMachine = i2
				if i2==rank:
					continue
				else:
					ltemp4=ltemp1[:]
					ltemp4.remove(i2)
					ltemp2=tuple(ltemp4)
					i3=fidWSet[ltemp2]
					
					keytemp=np.array(inputPartitionCollection[i3,i2][1],dtype='i')
					valuetemp=np.array(inputPartitionCollection[i3,i2][2],dtype='d')
					tempsize=keytemp.size
					inputPartitionCollection[i3,i2]=None
					maxnumber=tempsize/r
					
					# for first chunk to second last chunk
					for i4 in range(r-1):
						encodePreData[(i1,i3,i2,ltemp2[i4],1)]=keytemp[i4*maxnumber:(i4+1)*maxnumber]
						encodePreData[(i1,i3,i2,ltemp2[i4],2)]=valuetemp[i4*maxnumber:(i4+1)*maxnumber]
					# last chunk

					i4=i4+1

					
					encodePreData[(i1,i3,i2,ltemp2[i4],1)]=keytemp[i4*maxnumber:]
					encodePreData[(i1,i3,i2,ltemp2[i4],2)]=valuetemp[i4*maxnumber:]
					
					maxsize=max(maxsize,(encodePreData[i1,i3,i2,rank,1]).size)
			
			encodeTempKey=np.zeros(maxsize,dtype='i')
			encodeTempValue=np.zeros(maxsize,dtype='d')
			
			for i2 in ltemp1: # going over the nodes in s-sized subset 
				if i2==rank:
					continue
				else:
					ltemp4=ltemp1[:]
					ltemp4.remove(i2)
					ltemp2=tuple(ltemp4)
					i3=fidWSet[ltemp2]
					tempsize=(encodePreData[(i1,i3,i2,rank,1)]).size
					encodeTempKey+=np.append(encodePreData[(i1,i3,i2,rank,1)],-1*np.ones(maxsize-tempsize,dtype='i'))
					encodeTempValue+=np.append(encodePreData[(i1,i3,i2,rank,2)],-1*np.ones(maxsize-tempsize,dtype='d'))
					
			dicTemp={} # this creates a reference to a new dictionary and not clear the contents at the dictionary, so do not clear content of dictionary
			dicTemp[1]=encodeTempKey.dumps()
			dicTemp[2]=encodeTempValue.dumps()
			dicTemp[3]=[len(dicTemp[1]),len(dicTemp[2])]
			encodeDataSend[i1]=dicTemp
			encodeTempKey=None
			encodeTempValue=None
			dicTemp=None
		else:
			continue
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)

	########################################### Shuffle #############################################
	workerComm.Barrier()
	# Subset s-sized by subset s-sized
	t0=time.time()
	i1=0
	for i11 in WSubsetS:
		i1+=1
		if rank in i11:
			ltemp1=tuple(i11)
			mgComm=multicastGroupMap[ltemp1] # communicator for multicast for i1
			# rankTemp=mgComm.Get_rank()
			for i2 in ltemp1: # activeId
				mgComm.Barrier()
				rootId=ltemp1.index(i2)
				if rank==i2:
					data1=mgComm.bcast(encodeDataSend[i1][3],rootId)
					mgComm.Bcast([encodeDataSend[i1][1],MPI.CHAR],root=rootId)
					mgComm.Bcast([encodeDataSend[i1][2],MPI.CHAR],root=rootId)
					encodeDataSend[i1]=None
				else:
					temp=mgComm.bcast(None,rootId)
					buf1 = array('c', '\0') * temp[0]
					mgComm.Bcast([buf1,MPI.CHAR],root=rootId)
					enDataRecv1[i1,i2][1]=buf1[:temp[0]].tostring()

					buf2 = array('c', '\0') * temp[1]
					mgComm.Bcast([buf2,MPI.CHAR],root=rootId)
					enDataRecv1[i1,i2][2]=buf2[:temp[1]].tostring()
		else:
			continue
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)
	mgComm=None

	######################################### Unpack Stage ########################################
	workerComm.Barrier()
	t0=time.time()
	for i1 in enDataRecv1:
		enDataRecv[i1][1]=np.loads(enDataRecv1[i1][1])
		enDataRecv[i1][2]=np.loads(enDataRecv1[i1][2])
		enDataRecv[i1][3]=np.size(enDataRecv[i1][2])
		enDataRecv1[i1][1]=None
		enDataRecv1[i1][2]=None

	'''	######################### Clear memory ############################3
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)
	enDataRecv1.clear()
	'''

	########################################## Decoding ############################################
	# Obtain local values from the files in memory
	# append function cannot combine two lists
	workerComm.Barrier()
	t0=time.time()
	counti1=0
	for i1 in fid:
		ltemp1=WSubsetR[i1]
		if rank in ltemp1:
			counti1+=1
			keyDict[counti1]=inputPartitionCollection[i1,rank][1]
			valueDict[counti1]=inputPartitionCollection[i1,rank][2]
		else: 
			continue
	t01=time.time()-t0

	################################### Clear inputPartitionCollection if necessary ##########################
	inputPartitionCollection=None
	# gc.collect()
	# Obtain values from other machines
	i1=0
	t0=time.time()
	for i11 in WSubsetS:
		i1+=1
		if rank in i11:
			ltemp1=i11
			for i2 in ltemp1: # the active Id
				if i2==rank:
					continue
				else:
					tempsize=enDataRecv[(i1,i2)][3]
					encodeTempKey=enDataRecv[(i1,i2)][1]
					encodeTempValue=enDataRecv[(i1,i2)][2]
					for i3 in ltemp1: # chunk destid
						if i3==rank or i3==i2:
							continue
						else:
							ltemp2=i11[:]
							ltemp2.remove(i3)
							fidTemp=fidWSet[tuple(ltemp2)]
							tempsize2=(encodePreData[(i1,fidTemp,i3,i2,1)]).size
							encodeTempKey-=np.append(encodePreData[(i1,fidTemp,i3,i2,1)],-1*np.ones(tempsize-tempsize2,dtype='i'))
							encodeTempValue-=np.append(encodePreData[(i1,fidTemp,i3,i2,2)],-1*np.ones(tempsize-tempsize2,dtype='d'))
					# note that an issue can be there if encodeTemp or encodeValue is of size 1
					ltemp5=encodeTempKey.tolist()
					ltemp6=encodeTempValue.tolist()
					counti1+=1
					keyDict[counti1]=ltemp5
					valueDict[counti1]=ltemp6
		else:		
			continue
	t1=time.time()-t0+t01
	recvmsg1 = comm.gather(t1, root=0)

	################################### Reduce Phase ####################################
	workerComm.Barrier()
	t0=time.time()
	for kcount in keyDict:
		ltemp1=keyDict[kcount]
		ltemp2=valueDict[kcount]
		for i1,i2 in itertools.izip(ltemp1,ltemp2):
			if i1==-1:
				break
			PrankReduce[i1]+=i2
	
	# Transfer from reducers to mappers
	# print PrankReduce
	for i1 in PrankReduce:
		fidTemp=i1/g+1
		ltemp1=WSubsetR[fidTemp]
		for i2 in ltemp1:
			destPrank[i2-1][i1]=PrankReduce[i1]
	ltemp1=[x for x in range(1,K+1)]
	for i1 in ltemp1: # the active id
		if rank==i1: # send
			for i2 in ltemp1:
				if i2==rank:
					continue
				else:
					workerComm.send(destPrank[i2-1], dest=i2-1)
		else: # receive
			srcPrank[i1-1]=workerComm.recv(source=i1-1)
	# update PrankMap
	for i1 in ltemp1:
		if i1==rank:
			for i2 in destPrank[i1-1]:
				PrankMap[i2]=destPrank[i1-1][i2]
		else:
			for i2 in srcPrank[i1-1]:
				PrankMap[i2]=srcPrank[i1-1][i2]
	t1=time.time()-t0
	recvmsg1 = comm.gather(t1, root=0)


