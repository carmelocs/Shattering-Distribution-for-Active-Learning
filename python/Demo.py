import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.metrics.pairwise import rbf_kernel

# def rbf_kernel(X, Y, sigma):
#     N, K = X.shape
#     M = Y.shape[0]

#     K_xy = np.ones(M)*np.sum(X**2) + np.ones(N)*np.sum(Y**2) - 2*np.dot(X, Y.transpose())
#     K_xy = np.exp(-0.5 * K_xy / sigma**2)

#     return K_xy

def halving(K, m, candidate_index=None, lambda_=0.001):
    
    n = K.shape[0]
    # print(f'number of data: {n}')

    m = min(n, m)
    # print(f'number of samples: {m}')

    if candidate_index is None:
        candidate_index = np.array(range(n))
    
    # print(f'candidate_index: {candidate_index}')

    q = len(candidate_index)

    index = np.empty(m, dtype=int)
    # print(f'number of index: {index.shape}')

    print('Selecting samples......')
    for i in range(m):
        score = np.zeros(q)
        for j in range(q):
            k = candidate_index[j]
            # print(k)
            score[j] = np.dot(K[k, :], K[:, k]) / (K[k, k] + lambda_)
        
        I = score.argmax()
        # print(I)
        index[i] = candidate_index[I]

        # update K
        # K = K - np.dot(K[:, index[i]], K[index[i], :]) / (K[index[i], index[i]] + lambda_)
        K = K - K[:, index[i]][:, np.newaxis] @ K[index[i], :][np.newaxis, :] / (K[index[i], index[i]] + lambda_)

    print('Done.\n')
    return index

def number_density(data, center, radius):

    # print(f'length of data: {len(data)}\nlength of center: {len(center)}')

    f = 0
    for i in range(len(data)):
        ball_dist = np.zeros(len(center))
        dist = np.ones(len(center))
        for j in range(len(center)):
            dist[j] = np.linalg.norm(data[i, :] - center[j, :])
            if dist[j] < radius:
                ball_dist[j] = dist[j]

        # print(np.exp(ball_dist/1.8))
        f += np.sum(np.exp(ball_dist/1.8)**2) / (len(ball_dist) + 1)
    
    return f

def SDAL(data, k):

    kmeans = KMeans(n_clusters=k).fit(data)
    center = kmeans.cluster_centers_

    radius = 0.25
    L, R = data.shape
    
    f = number_density(data, center, radius)
    T = 0
    while T<50:
        for j in range(k):
            ball = []
            dist = np.empty(L)
            for i in range(L):
                dist[i] = np.linalg.norm(data[i] - center[j])
                if dist[i] < radius:
                    ball.append(data[i])
            if len(ball)==0:
                center[j] = center[j]
            else:
                center[j] = np.mean(ball)

        F = number_density(data, center, radius)
        
        if F-f==0 or len(np.argwhere(pdist(center)<2*radius))>0:
            break
        else:
            f = F
        T+=1
        radius*=1.1
    
    tree = KDTree(data)
    _, idx = tree.query(center, k=1)
    # print(idx)
    center = data[idx].squeeze()
            
    return center


if __name__ == '__main__':
    import scipy.io

    test = False

    # np.random.seed(0)
    # data = np.random.rand(10, 2)
    # print(f'min of data: {data.min()}')
    # print(f'max of data: {data.max()}')

    mat = scipy.io.loadmat('Syndata.mat') 
    data = mat['data']

    K = rbf_kernel(data, data, gamma=1.8**(-2))

    id = halving(K,400)

    X = np.random.rand(400, 2) if test else data[id]

    # print(data[0])
    # print(np.unique(id, return_counts=True))
    # print(X.shape)
    # print(np.unique(X, return_counts=True))
    print(len(set(id)))

    center = SDAL(X,4)
    print(center)
    print(center.shape)
    print(len(set(center[:,1])))