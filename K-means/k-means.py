from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(0)
df=pd.read_csv("data.csv",delimiter="\t")

f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X=np.array([f1,f2]).T

X /= X.max(axis=0, keepdims=True)


print(X.shape)


#X, y = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, n_features=2,random_state=42)

max_iterations=100
def init_centroids():
    centroids=np.zeros((k,X.shape[1]))
    for i in range(k):
        centroid = X[np.random.choice(range(X.shape[0]))]
        centroids[i]=centroid
    return centroids

def getdistance(sample,centroid):
    return np.sqrt(np.sum(np.square(sample-centroid)));


def closest_centroid(point,centroids):
    closest_i = 0
    closest_dist = float('inf')
    for i, centroid in enumerate(centroids):
        distance = getdistance(point, centroid)
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i

def create_clusters(centroids):
    clusters=[[] for _ in range(k)]
    for i, sample in enumerate(X):
        centroid_i=closest_centroid(sample,centroids)
        clusters[centroid_i].append(i)
    return clusters


def calculate_centroids(clusters):
        
    n_features = np.shape(X)[1]
    centroids = np.zeros((k, n_features))
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids


def get_cluster_labels(clusters):
        
    y_pred = np.zeros(np.shape(X)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred


def predict():
    
    centroids = init_centroids()

        
    for _ in range(max_iterations):
            
        clusters = create_clusters(centroids)
            
        prev_centroids = centroids
            
        centroids = calculate_centroids(clusters)
            
        diff = centroids - prev_centroids
        if not diff.any():
            print("Algorithm-Converged")
            break

    return clusters,get_cluster_labels(clusters), centroids



listofk=[]

for i in range(1,10): 
    k=i
    avdistance=np.zeros((k))
    clusters,ypre,centroids=predict()

    for i,cluster in enumerate(clusters):
        for point in cluster:
            avdistance[i] += getdistance(point,centroids[i])
    avdistance = avdistance.sum()/k ;
    listofk.append(avdistance)


k=4
clusters,ypre,centroids=predict()
print(ypre.shape)
plt.subplot(221)

plt.scatter(X[:,0],X[:,1],c=ypre)
for i in range(k):
    plt.scatter(centroids[i,0],centroids[i,1],marker='*',c='b',s=60)
plt.subplot(222)
plt.plot(np.arange(1,10,1),listofk,marker='.')
plt.xlabel("K")
plt.ylabel("Average Within-cluster Distance")



plt.show()
