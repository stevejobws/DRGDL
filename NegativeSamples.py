 #encoding:utf-8
 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

dataset = 'F-Dataset'
DDAs = pd.read_csv('./data/'+dataset+'/DiDrA.txt',sep='\t',header=None).values
whole_positive_index=[]
whole_negative_index=[]
for i in range(np.shape(DDAs)[0]):
    for j in range(np.shape(DDAs)[1]):
        if int(DDAs[i][j])==1:
            whole_positive_index.append([i,j])
        elif int(DDAs[i][j])==0:
            whole_negative_index.append([i,j])
        else: print ('wrong')
DrugSim = pd.read_csv('./data/'+dataset+'/DrugSim.txt',sep='\t', header=None)
DiseaseSim = pd.read_csv('./data/'+dataset+'/DiseaseSim.txt',sep='\t', header=None)
Positive = pd.DataFrame(whole_positive_index)
Negative = pd.DataFrame(whole_negative_index)
Positive[2] = Positive.apply(lambda x:1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x:0 if x[0] < 0 else 0, axis=1)
results = pd.concat([Positive,Negative]).reset_index(drop=True)
X = pd.concat([DiseaseSim.loc[results[0].values.tolist()].reset_index(drop=True),DrugSim.loc[results[1].values.tolist()].reset_index(drop=True)],axis=1)
Y = results[2]


def clustering(Z):
    
    n_clusters = 30 # C-Dataset 50 F-Dataset 30
    model = KMeans(n_clusters=n_clusters,init='k-means++',n_init=10,max_iter=300,tol=0.0001,random_state=0,algorithm ='auto')
    cluster_id = model.fit_predict(Z)
    
    clusters = []
    for i in range(n_clusters):
        clusters.append([j for j, x in enumerate(cluster_id) if x == i])
    
    new_negetivesample = pd.DataFrame()
    for i in range(n_clusters): 
        if len(np.asarray(clusters[i])[np.asarray(clusters[i]) < len(whole_positive_index)]) < 20: 
            new_negetivesample = new_negetivesample.append(results.loc[(np.asarray(clusters[i])[np.asarray(clusters[i]) > len(whole_positive_index)]).tolist()])
    print("new_negetivesample",len(new_negetivesample))
    new_negetivesample.sample(n=len(whole_positive_index)).to_csv('./data/'+dataset+'/Newnegetivesample.csv', header=0,index=0)
    return (model.labels_, model.cluster_centers_)

x = np.array([[1.0, 1.0, 1.0], 
              [2.0, 3.0, 2.0],
              [3.0, 3.0, 2.0],
              [1.0, 3.0, 5.0],
              [2.0, 3.0, 6.0],
              [2.0, 1.0, 2.0],
              [2.0, 4.0, 2.0],
              [6.0, 3.0, 2.0],
              [2.0, 0.0, 1.0],
              [8.0, 3.0, 2.0],
              ])

clustering(X)
