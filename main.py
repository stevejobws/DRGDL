 #encoding:utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc,average_precision_score
from scipy import interp
from scipy.sparse import identity
import torch
from torch.autograd import Variable
from src.gat import *
from src.utils import *
from src.processing import *

def GAT_F(dataset):
    # Training settings
    learning_rate = 0.01
    weight_decay = 5e-4
    epoch_num = 200
    dropout = 0.02
    #in_size = node_features 
    hi_size = 64 # 16 
    alpha = 0.2
    n_heads = 4
    name = locals() 
    for i in range(10):
        Adj = load_file_as_Adj_matrix('./data/'+dataset+'/DTI_train'+str(i)+'.csv')
        train_features = load_data(features)       
        I = identity(features.shape[0]).toarray()        
        adj = torch.from_numpy(Adj)
        adj = normalize(adj + torch.from_numpy(I))
        adj = torch.from_numpy(adj)
        model = GAT(n_feat=features.shape[1],
                n_hid=hi_size,
                n_class=64,#labels.max() + 1,
                dropout=dropout,
                alpha=alpha, 
                n_heads=n_heads)
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0')
        
        train_features, adj = Variable(train_features), Variable(adj)
        if torch.cuda.is_available():     
            model.cuda(device)  
            train_features = train_features.cuda(device)
            adj = adj.cuda(device)           
        output, Emdebding_train = model(train_features, adj)
        Emdebding_GAT = pd.DataFrame(Emdebding_train.cpu().detach().numpy())
        Emdebding_GAT.to_csv('./data/'+dataset+'/Emdebding_GAT'+str(i)+'.csv', header=None,index=False)
        print(i)

def main(dataset,data_train,data_test,labels_train,labels_test):
        
    print("10-fold cross validation")
    tprs=[]
    aucs=[]
    mean_fpr=np.linspace(0,1,1000)

    for i in range(10):
        X_train,X_test = data_train[i],data_test[i]
        Y_train,Y_test = np.array(labels_train[i]),np.array(labels_test[i])
        best_RandomF = RandomForestClassifier(n_jobs=-1)# n_estimators=999,
        best_RandomF.fit(np.array(X_train), np.array(Y_train))
        y_score0 = best_RandomF.predict(np.array(X_test))
        y_score_RandomF = best_RandomF.predict_proba(np.array(X_test))
       
        fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0]=0.0
        #auc
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        average_precision = average_precision_score(Y_test, y_score_RandomF[:,1])
        print("fold = {} auc = {} aupr = {} ".format(i, roc_auc,average_precision))

    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc=auc(mean_fpr,mean_tpr)
    print("mean_auc:",mean_auc)        

if __name__ == '__main__':
    
    import optparse
    import sys
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-d", "--dataset", action='store',
                      dest='dataset', default='F-Dataset', type='str',
                      help=('The benchmark dataset contains F-Dataset and C-Dataset'))
    options, args = parser.parse_args()   
    dataset = args[0]
    print(dataset)
    
    # produce Gat lower-order representations
    GAT_F(dataset)
    # train data
    creat_var = locals()
    creat_var = locals()
    Negative = pd.read_csv('./data/'+dataset+'/Newnegetivesample.csv',header=None) # NegativeSample Newnegetivesample
    Negative[1] = Negative[1]+Negative[0].max()+1
    Nindex = pd.read_csv('./data/'+dataset+'/NewRandomList.csv',header=None)
    Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
    for i in range(10):
        Embedding_GAT = pd.read_csv('./data/'+dataset+'/Emdebding_GAT'+str(i)+'.csv',header=None)
        df = pd.DataFrame(np.random.randint(0,1,size=(Embedding_GAT.shape)))
        Embedding_Node2vec = pd.read_csv('./data/'+dataset+'/Embedding_Node2vec'+str(i)+'.txt', sep=' ',header=None,skiprows=1)
        Embedding_Node2vec = Embedding_Node2vec.sort_values(0,ascending=True)
        Embedding_Node2vec.set_index([0], inplace=True)
        Embedding_Node2vec = df.add(Embedding_Node2vec, fill_value = 0)
        Embedding_Node2vec = Embedding_Node2vec.fillna(0)
        Embedding_Node2vec = Embedding_Node2vec.iloc[:,1:]
        train_data = pd.read_csv('./data/'+dataset+'/DTI_train'+str(i)+'.csv',header=None)
        train_data[2] = train_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
        kk = []
        for j in range(10):
            if j !=i:
                kk.append(j)
        index = np.hstack([np.array(Nindex)[kk[0]],np.array(Nindex)[kk[1]],np.array(Nindex)[kk[2]],np.array(Nindex)[kk[3]],np.array(Nindex)[kk[4]],
                           np.array(Nindex)[kk[5]],np.array(Nindex)[kk[6]],np.array(Nindex)[kk[7]],np.array(Nindex)[kk[8]]])
        result = train_data.append(pd.DataFrame(np.array(Negative)[index]))    
        labels_train = result[2]
        data_train_feature = pd.concat([pd.concat([Embedding_GAT.loc[result[0].values.tolist()],Embedding_Node2vec.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
                                        pd.concat([Embedding_GAT.loc[result[1].values.tolist()],Embedding_Node2vec.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)
        creat_var['data_train'+str(i)] = data_train_feature
        creat_var['labels_train'+str(i)] = labels_train
        print(len(labels_train))
        del labels_train, result, data_train_feature
        test_data = pd.read_csv('./data/'+dataset+'/DTI_test'+str(i)+'.csv',header=None)
        test_data[2] = test_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
        result = test_data.append(pd.DataFrame(np.array(Negative)[np.array(Nindex)[i]]))    
        labels_test = result[2]
       
        data_test_feature = pd.concat([pd.concat([Embedding_GAT.loc[result[0].values.tolist()],Embedding_Node2vec.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
                                       pd.concat([Embedding_GAT.loc[result[1].values.tolist()],Embedding_Node2vec.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)
        creat_var['data_test'+str(i)] = data_test_feature.values.tolist()
        creat_var['labels_test'+str(i)] = labels_test.values.tolist()
        print(len(labels_test))
        del train_data, test_data, labels_test, result, data_test_feature, df   
        print(i)
        print(np.array(data_train0).shape)
        
    data_train = [data_train0,data_train1,data_train2,data_train3,data_train4,data_train5,data_train6,data_train7,data_train8,data_train9]
    data_test = [data_test0,data_test1,data_test2,data_test3,data_test4,data_test5,data_test6,data_test7,data_test8,data_test9]
    labels_train = [labels_train0,labels_train1,labels_train2,labels_train3,labels_train4,labels_train5,labels_train6,labels_train7,labels_train8,labels_train9]
    labels_test = [labels_test0,labels_test1,labels_test2,labels_test3,labels_test4,labels_test5,labels_test6,labels_test7,labels_test8,labels_test9]
    print(np.array(data_train0).shape)
    print(np.array(data_test0).shape)
    print(np.array(labels_train0).shape)
    print(np.array(labels_test0).shape)
    
    main(dataset,data_train,data_test,labels_train,labels_test)

    