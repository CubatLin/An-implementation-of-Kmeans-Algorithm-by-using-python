##Kmeans algorithm Source Code by MIACadm
#--------------------------
#sample data:iris 
#--------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random 

iris = datasets.load_iris()
N=3#GroupNumbers
data = pd.DataFrame(iris.data, columns = iris['feature_names'])
#random centers
centers_unif = data.sample(N)+random.uniform(0,0.8) #+ random.uniform(1,2)
centers_unif.index=range(N)
#euclidean
dis_list=[]
for i in range(len(data)):
    for j in range(N):
        dis_list.append(np.sqrt(sum((data.iloc[i]-centers_unif.iloc[j])**2)))
#Which group
whichmin_list=[]
for i in np.arange(0,len(dis_list),N):
    temp=dis_list[i:i+N]
    whichmin_list.append(temp.index(min(temp)))
data['GroupFlag']=whichmin_list
data.groupby('GroupFlag').size()
#每一個group的mean
centers_mean = pd.DataFrame()
for i in range(N):
    centers_mean[i]=data[(data["GroupFlag"] == i)].mean()
centers_mean = centers_mean.drop('GroupFlag')
centers_mean = centers_mean.T
plt.style.use('ggplot')
plt.figure(figsize=(10,5.5))
plt.subplot(1,2,1)
plt.axis([0,3,0.5,7.5])
plt.scatter(data['petal width (cm)'] , data['petal length (cm)'],c=data['GroupFlag'])
plt.plot(centers_mean['petal width (cm)'],centers_mean['petal length (cm)'],'*',color ='red')
plt.title('Kmeans Algorithm - Before',fontsize='large',fontweight='bold') 
plt.xlabel('petal width (cm)') 
plt.ylabel('petal length (cm)')

#####迭代######
#以平均組中點更新:centers_mean
    #1 euclidean
count=0
while count<20:
    data = data.drop(columns=['GroupFlag'])
    dis_list=[]
    for i in range(len(data)):
        for j in range(N):
            dis_list.append(np.sqrt(sum((data.iloc[i]-centers_mean.iloc[j])**2)))
    #2 Which group
    whichmin_list=[]
    for i in np.arange(0,len(dis_list),N):
        temp=dis_list[i:i+N]
        whichmin_list.append(temp.index(min(temp)))
    data['GroupFlag']=whichmin_list
        #3每一個group的mean
    centers_mean = pd.DataFrame()
    for i in range(N):
        centers_mean[i]=data[(data["GroupFlag"] == i)].mean()
    centers_mean = centers_mean.drop('GroupFlag')
    print(centers_mean) #輸出調整組中點
    centers_mean = centers_mean.T
    count = count+1

plt.subplot(1,2,2)
plt.axis([0,3,0.5,7.5])
plt.scatter(data['petal width (cm)'] , data['petal length (cm)'],c=data['GroupFlag'])
plt.plot(centers_mean['petal width (cm)'],centers_mean['petal length (cm)'],'*',color ='red')
plt.title('Kmeans Algorithm - After',fontsize='large',fontweight='bold') 
plt.xlabel('petal width (cm)')
plt.ylabel('petal length (cm)')
plt.show()
   
data.groupby('GroupFlag').size()
