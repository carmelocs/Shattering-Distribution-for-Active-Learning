import numpy as np
import torch


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

if __name__ == '__main__':
    import scipy.io
    from sklearn.metrics.pairwise import rbf_kernel
    import matplotlib.pyplot as plt

    mat = scipy.io.loadmat('Syndata.mat') 
    data = mat['data']

    K = rbf_kernel(data, data, gamma=0.5*1.8**(-2))

    idx = halving(K, n_samples=400)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(data[:, 0], data[:, 1], color='b')
    ax1.set_title(f'Input data')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(data[idx, 0], data[idx, 1], color='g')
    ax2.set_title(f'Shattered data')
    
    plt.show()