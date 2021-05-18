import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.metrics.pairwise import rbf_kernel

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# def rbf_kernel(X, Y, sigma):
#     N, K = X.shape
#     M = Y.shape[0]

#     K_xy = np.ones(M)*np.sum(X**2) + np.ones(N)*np.sum(Y**2) - 2*np.dot(X, Y.transpose())
#     K_xy = np.exp(-0.5 * K_xy / sigma**2)

#     return K_xy

def halving(K, n_samples, candidate_index=None, lambda_=0.001):
    
    n_data = K.shape[0]
    # print(f'number of data: {n}')

    n_samples = min(n_data, n_samples)
    # print(f'number of samples: {m}')

    if candidate_index is None:
        candidate_index = np.array(range(n_data))
    
    # print(f'candidate_index: {candidate_index}')

    n_query = len(candidate_index)

    index = np.empty(n_samples, dtype=int)
    # print(f'number of index: {index.shape}')

    # print('Selecting samples......')
    for i in range(n_samples):
        score = np.zeros(n_query)
        for j in range(n_query):
            if candidate_index[j] == -1:
                continue
            else:
                k = candidate_index[j]
                score[j] = np.dot(K[k, :], K[:, k]) / (K[k, k] + lambda_)
        
        I = score.argmax()
        index[i] = candidate_index[I]

        candidate_index[I] = -1
        
        # update K
        K = K - K[:, index[i]][:, np.newaxis] @ K[index[i], :][np.newaxis, :] / (K[index[i], index[i]] + lambda_)

    # print('Done.\n')
    return index

def number_density(data, center, radius):
    # print(f'length of data: {len(data)}\nlength of center: {len(center)}')
    f = 0
    for i in range(len(data)):
        ball_dist = []
        dist = np.zeros(len(center))
        for j in range(len(center)):
            dist[j] = np.linalg.norm(data[i, :] - center[j, :])
            if dist[j] < radius:
                ball_dist.append(dist[j])

        # print(np.exp(ball_dist/1.8))
        f += np.sum(np.exp(np.array(ball_dist)/1.8)**2) / (len(ball_dist) + 1)
    
    return f


def SDAL(data, k):

    kmeans = KMeans(n_clusters=k).fit(data)
    center = kmeans.cluster_centers_

    radius = 0.25
    num_data, _ = data.shape
    
    f = number_density(data, center, radius)

    for T in range(50):
        for j in range(k):
            ball = []
            dist = np.zeros(num_data)
            for i in range(num_data):
                dist[i] = np.linalg.norm(data[i, :] - center[j, :])
                if dist[i] < radius:
                    ball.append(data[i])
            # print(len(ball))
            if len(ball)==0:
                center[j] = center[j]
            else:
                center[j] = np.mean(np.array(ball), axis=0)
                # print(center[j])

        F = number_density(data, center, radius)

        if (F-f)==0 or len(np.argwhere(pdist(center)<2*radius))>0:
            break
        else:
            f = F
        radius*=1.1
    
    tree = KDTree(data)
    _, idx = tree.query(center, k=1)
    # print(idx)
    center = data[idx].squeeze()
            
    return center, radius


if __name__ == '__main__':
    import scipy.io

    test = False

    # np.random.seed(0)
    # data = np.random.rand(10, 2)
    # print(f'min of data: {data.min()}')
    # print(f'max of data: {data.max()}')

    mat = scipy.io.loadmat('Syndata.mat') 
    data = mat['data']

    K = rbf_kernel(data, data, gamma=0.5*1.8**(-2))

    # id = halving(K,400)

    indices = []
    for i in range(1, 9):
        idx = halving(K, 100*i)
        indices.append(idx)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    # for i, idx in enumerate(indices, 1):
    #     ax = fig.add_subplot(2,3,i)
    #     ax.scatter(data[idx, 0], data[idx, 1], color='g', s=10, label='samples')
    #     ax.legend()
    #     print(f'Number of samples in iter_{i}: {len(set(idx))}')
    # plt.show()

    for i, idx in enumerate(indices, 1):
        
        num_samples = 100*i

        X = data[idx]
        center, radius = SDAL(X,5)
        # print(f'center: \n{center}\n{center.shape}')
        # print(len(set(center[:,1])))
        print(f'Radius of iter_{i}: {radius}')
            
        ax = fig.add_subplot(2, 4, i)
        ax.scatter(data[idx, 0], data[idx, 1], color='b', s=10)
        ax.scatter(center[:, 0], center[:, 1], color='g', marker='s', s=10)
        ax.scatter(center[:, 0], center[:, 1], color='', marker='o', edgecolors='r', s=4000*radius)
        plt.title(f'{num_samples} samples (radius: {radius})')
    
    plt.show()


    # print(data[0])
    # print(np.unique(id, return_counts=True))
    # print(X.shape)
    # print(np.unique(X, return_counts=True))
    # print(f'number of samples: {len(set(id))}')




    