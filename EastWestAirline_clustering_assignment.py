#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage


# In[2]:


flight=pd.read_csv('C:/Users/vinay/Desktop/Clustering/eastwestairlines/EastWestAirlines.csv')


# In[3]:


flight


# In[4]:


#Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[5]:


df_norm= norm_func(flight.iloc[:,1:])
df_norm


# In[6]:


df_norm.describe()


# In[7]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='complete'))


# In[8]:


#create clusters
cluster=AgglomerativeClustering(n_clusters=14,affinity="euclidean",linkage="complete").fit(df_norm)
cluster


# In[9]:


# save clusters for chart
h=pd.Series(cluster.labels_)
flight['clust']=h


# In[10]:


flight1=flight.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
flight1


# In[11]:


flight2= flight1.iloc[:,2:].groupby(flight1.clust).median()
flight2


# # Kmeans

# In[12]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[13]:


k=list(range(10,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[14]:


TWSS


# In[15]:


plt.plot(k,TWSS, 'ro-')
plt.xlabel('number of clusters')
plt.ylabel('total within sum of squares')
plt.xticks(k)


# ## Selecting 14 clusters from the above scree plot which is the optimum number of clusters

# In[16]:


model1=KMeans(n_clusters=14)
model1.fit(df_norm)


# In[17]:


# getting the labels of clusters assigned to each row 
model1.cluster_centers_
model1.labels_
model=pd.Series(model1.labels_)# converting numpy array into pandas series object 
model


# In[18]:


# creating a  new column and assigning it to new column 
flight['clust']=model
flight


# In[19]:


flightfinal=flight.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]


# In[20]:


Airlines=flight.iloc[:,1:13].groupby(flightfinal.clust).mean();Airlines


# # DBSCAN

# In[23]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[24]:


array=flight.values;array


# In[25]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[26]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[27]:


dbscan.labels_


# In[29]:


DBfinal=pd.DataFrame(dbscan.labels_,columns=['cluster'])
DBfinal


# In[30]:


pd.concat([flight,DBfinal],axis=1)


# In[ ]:




