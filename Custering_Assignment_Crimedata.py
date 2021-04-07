#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[2]:


crime_1 = pd.read_csv("C:/Users/vinay/Downloads/crime_data.csv")


# In[3]:


crime_1


# In[4]:


#Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[5]:


df_norm = norm_func(crime_1.iloc[:,1:]);df_norm


# In[6]:


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(df_norm,method='complete',metric='euclidean')


# In[7]:


plt.figure(figsize=(15,5))
plt.title('Hierarchical Clustering Dendogram')
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=8) 
plt.show()


# In[8]:


#create clusters
h_clustering=AgglomerativeClustering(n_clusters=6,affinity="euclidean",linkage="complete").fit(df_norm)
h_clustering


# In[9]:


h=pd.Series(h_clustering.labels_)


# In[10]:


crime_1['clust']=h
crime_1=crime_1.iloc[:,[5,0,1,2,3,4]]
crime_1


# In[11]:


crime_1.iloc[:,2:].groupby(crime_1.clust).median()


# # Kmeans

# In[12]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[13]:


###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[14]:


# Scree plot 
plt.plot(k,TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS");
plt.xticks(k)


# In[15]:


# Selecting 3 clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=3).fit(df_norm)
model.cluster_centers_


# In[25]:


# getting the labels of clusters assigned to each row 
model.labels_ 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime_1['clust']=md # creating a  new column and assigning it to new column 
crime_1.head()


# In[28]:


crime_1.iloc[:,1:5].groupby(crime_1.clust).mean() ;crime_1


# # DBSCAN

# In[20]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[21]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[29]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[36]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[37]:


pd.concat([crime_1,cl],axis=1)


# In[ ]:





# In[ ]:




