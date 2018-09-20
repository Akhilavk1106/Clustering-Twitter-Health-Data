
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets
import pandas as pd
import sys
original_labels=np.empty(16,dtype='int64')
tweet_num=0
tweets=[]
listfile = []
for file_name in sys.argv[1:]:
    list=[]
    with open(file_name,encoding='ISO-8859-1') as file:
        list_special=['rt','video','amp','may']
        for row in file.readlines():
            content=row.split('|')
            c=content[-1].split(' http')
            c[0]=c[0].lower()
            remove_pun = re.sub("[\s+\.\-\!\;\:\/_,$%^(+\"\']+|[+——！，。?、~@#￥%…&…*（）]+", " ", c[0])
            list.append(remove_pun)

    tweets.extend(list)
    original_labels[tweet_num]=len(list)
    tweet_num +=1

print(original_labels)

true_cluster2 = np.empty((len(tweets), 1), dtype='int64')
begin = 0
for i in range(len(original_labels)):
    end = begin + original_labels[i]
    true_cluster2[begin:end, 0] = i
    begin = end

vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
X = vectorizer.fit_transform(tweets)
array_trans=X.toarray()


pca2=PCA(n_components=2)
newMat = pca2.fit_transform(array_trans)
kmeans = KMeans(n_clusters=16,random_state=0).fit(newMat)
labels = kmeans.labels_


X_clustered = kmeans.fit_predict(newMat)

ind=0
print(true_cluster2)
for la in labels:
    print('OriginalLabel:',true_cluster2[ind],'ClusterLabel',la)
    ind+=1

#Define our own color map
LABEL_COLOR_MAP = {0: 'b', 1: 'c', 2: 'k',3:'m', 4: 'green', 5: 'r',6:'w', 7: 'y', 8: 'ivory',9:'navy', 10: 'orange', 11: 'purple',12:'olive', 13: 'gray', 14: 'maroon',15:'pink', 16: 'tan'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# Plot the scatter digram
plt.figure(figsize = (25,25))
plt.scatter(newMat[:,0],newMat[:,1], c= label_color, alpha=0.5)
plt.show()




