##Kmeans algorithm Source Code by MIACadm
#--------------------------
#sample data:iris 
#--------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random 

def Kmeans_SC(dataset,K):
    centers_unif = dataset.sample(N)+random.uniform(0,0.8) #+ random.uniform(1,2)
    centers_unif.index=range(N)
    #euclidean
    dis_list=[]
    for i in range(len(dataset)):
        for j in range(N):
            dis_list.append(np.sqrt(sum((dataset.iloc[i]-centers_unif.iloc[j])**2)))
    #Which group
    whichmin_list=[]
    for i in np.arange(0,len(dis_list),N):
        temp=dis_list[i:i+N]
        whichmin_list.append(temp.index(min(temp)))
    dataset['GroupFlag']=whichmin_list
    dataset.groupby('GroupFlag').size()
    #每一個group的mean
    centers_mean = pd.DataFrame()
    for i in range(N):
        centers_mean[i]=dataset[(dataset["GroupFlag"] == i)].mean()
    centers_mean = centers_mean.drop('GroupFlag')
    centers_mean = centers_mean.T
    plt.style.use('ggplot')
    plt.figure(figsize=(10,5.5))
    plt.subplot(1,2,1)
    plt.axis([0,3,0.5,7.5])
    plt.scatter(dataset['petal width (cm)'] , dataset['petal length (cm)'],c=dataset['GroupFlag'])
    plt.plot(centers_mean['petal width (cm)'],centers_mean['petal length (cm)'],'*',color ='red')
    plt.title('Kmeans Algorithm - Before',fontsize='large',fontweight='bold') 
    plt.xlabel('petal width (cm)') 
    plt.ylabel('petal length (cm)')
    
    
    #####迭代######
    #以平均組中點更新:centers_mean
        #1 euclidean
    count=0
    c_logitd=1
    while c_logitd>0.000000000001 and count<1000:
        dataset = dataset.drop(columns=['GroupFlag'])
        dis_list=[]
        for i in range(len(dataset)):
            for j in range(N):
                dis_list.append(np.sqrt(sum((dataset.iloc[i]-centers_mean.iloc[j])**2)))
        #2 Which group
        whichmin_list=[]
        for i in np.arange(0,len(dis_list),N):
            temp=dis_list[i:i+N]
            whichmin_list.append(temp.index(min(temp)))
        dataset['GroupFlag']=whichmin_list
        #3每一個group的mean
        centers_mean_2 = pd.DataFrame()
        for i in range(N):
            centers_mean_2[i]=dataset[(dataset["GroupFlag"] == i)].mean()
        centers_mean_2 = centers_mean_2.drop('GroupFlag')
        centers_mean_2 = centers_mean_2.T
        
        c_dis = (centers_mean_2-centers_mean)**2
        c_dis = c_dis.sum(axis=1)
        c_logitd = c_dis.sum(axis=0)
        print("Sum Distances of Centers Moving:")
        print(c_logitd)
        print("")
        print(centers_mean_2) #輸出調整組中點
        centers_mean = centers_mean_2
        count = count+1
        
    plt.subplot(1,2,2)
    plt.axis([0,3,0.5,7.5])
    plt.scatter(dataset['petal width (cm)'] , dataset['petal length (cm)'],c=dataset['GroupFlag'])
    plt.plot(centers_mean['petal width (cm)'],centers_mean['petal length (cm)'],'*',color ='red')
    plt.title('Kmeans Algorithm - After',fontsize='large',fontweight='bold') 
    plt.xlabel('petal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.show()
       
    return(dataset.groupby('GroupFlag').size())
    

#------------------------#   
#import sample data : iris
iris = datasets.load_iris()
N=3 #K Numbers
data = pd.DataFrame(iris.data, columns = iris['feature_names'])
Kmeans_SC(data,N)