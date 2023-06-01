 #encoding:utf-8
import numpy as np
import pandas as pd
import optparse
import sys

# LoadData
sys.path.append("..")
from main import *
print(' '.join (sys.argv))
dataset = sys.argv[1]
Cdata = pd.read_csv('./data/'+dataset+'/DiDrA.txt',sep='\t',header = None)
CDrug = pd.read_csv('./data/'+dataset+'/DrugSim.txt',sep='\t',header = None)
CDisease = pd.read_csv('./data/'+dataset+'/DiseaseSim.txt',sep='\t',header = None)

# AutoEncoder
from keras.layers import Dense,Activation,Input
from keras.models import Sequential,Model

## Drug feature reducing dimension
inputDims = CDrug.values.shape[1]
EncoderDims = 64

AutoEncoder = Sequential()
AutoEncoder.add(Dense(input_dim=inputDims,units=EncoderDims,activation='relu'))
AutoEncoder.add(Dense(input_dim=EncoderDims,units=inputDims,activation='sigmoid'))
AutoEncoder.compile(optimizer='adadelta',loss='binary_crossentropy')
AutoEncoder.fit(CDrug.values,CDrug.values,batch_size=32, epochs=50,shuffle=True)

# Output Features
from keras import backend as K
get_2rd_layer_output = K.function([AutoEncoder.layers[0].input], [AutoEncoder.layers[0].output])
new_CDrug = get_2rd_layer_output([CDrug.values])[0]

## Disease feature reducing dimension
inputDims = CDisease.values.shape[1]
EncoderDims = 64
AutoEncoder_CDisease = Sequential()

AutoEncoder_CDisease.add(Dense(input_dim=inputDims,units=EncoderDims,activation='relu'))
AutoEncoder_CDisease.add(Dense(input_dim=EncoderDims,units=inputDims,activation='sigmoid'))
AutoEncoder_CDisease.compile(optimizer='adadelta',loss='binary_crossentropy')
AutoEncoder_CDisease.fit(CDisease.values,CDisease.values,batch_size=32,epochs=50,shuffle=True)

# Output Features
get_2rd_layer_output = K.function([AutoEncoder_CDisease.layers[0].input], [AutoEncoder_CDisease.layers[0].output])
new_CDisease = get_2rd_layer_output([CDisease.values])[0]
new_CDisease

# Save feature
features = pd.concat([pd.DataFrame(new_CDrug),pd.DataFrame(new_CDisease)],axis=0)
features.reset_index(drop=True, inplace=True) 
features.to_csv('./data/'+dataset+'/features.csv', header=None,index=False)

# Produce associations
Adj = Cdata.values
rows,cols = Adj.shape 
DDI_drug = pd.DataFrame()
DDI_disease = pd.DataFrame()
for i in range(rows):# disease
    for j in range(cols): #drug
        if Adj[i][j] == 1:
            DDI_drug=DDI_drug.append([i],ignore_index=True)
            DDI_disease=DDI_disease.append([j+rows],ignore_index=True)
DDI = pd.concat([DDI_drug,DDI_disease],axis=1)
DDI.to_csv('./data/'+dataset+'/DrugDiseaseInteractions.csv', index=False)

# Produce random seed
import math
import random
def partition(ls, size):
    """
    Returns a new list with elements
    of which is a list of certain size.

        >>> partition([1, 2, 3, 4], 3)
        [[1, 2, 3], [4]]
    """
    return [ls[i:i+size] for i in range(0, len(ls), size)]
RandomList = random.sample(range(0, len(DDI)), len(DDI))
NewRandomList = partition(RandomList, math.ceil(len(RandomList) / 10))
NaN = pd.isnull(NewRandomList).any(0).nonzero()[0]
NewRandomList = pd.DataFrame(NewRandomList)
NewRandomList = NewRandomList.fillna(int(0))
NewRandomList = NewRandomList.astype(int)
NewRandomList.to_csv('./data/'+dataset+'/NewRandomList.csv', header=None,index=False)

# split train and test sets
Nindex = pd.read_csv('./data/'+dataset+'/NewRandomList.csv',header=None)
for i in range(len(Nindex)):
    kk = []
    for j in range(10):
        if j !=i:
            kk.append(j)
    index = np.hstack([np.array(Nindex)[kk[0]],np.array(Nindex)[kk[1]],np.array(Nindex)[kk[2]],np.array(Nindex)[kk[3]],np.array(Nindex)[kk[4]],
                       np.array(Nindex)[kk[5]],np.array(Nindex)[kk[6]],np.array(Nindex)[kk[7]],np.array(Nindex)[kk[8]]])
    DDI_train= pd.DataFrame(np.array(DDI)[index])
    DDI_train.to_csv('./data/'+dataset+'/DTI_train'+str(i)+'.csv', header=None,index=False)
    DDI_train = DDI_train.sample(frac=1.0)
    DDI_train.to_csv('./data/'+dataset+'/DTI_train'+str(i)+'.txt', sep='\t' ,header=None,index=False)
    DDI_test=pd.DataFrame(np.array(DDI)[np.array(Nindex)[i]])
    DDI_test.to_csv('./data/'+dataset+'/DTI_test'+str(i)+'.csv', header=None,index=False)
    print(i)
    
