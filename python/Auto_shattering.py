import numpy as np
from sqlalchemy.sql.expression import false
import torch
from sklearn.metrics.pairwise import rbf_kernel
import scipy.io
import random
import matplotlib.pyplot as plt


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
     source: 源域数据，行表示样本数目，列表示样本数据维度
     target: 目标域数据 同source
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
  sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    # 求矩阵的行数，即两个域的的样本总数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差 
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = ((total0-total1)**2).sum(2) 
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # print(bandwidth_list)
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
     source: 源域数据，行表示样本数目，列表示样本数据维度
     target: 目标域数据 同source
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
  loss: MMD loss
    '''
    source_num = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    target_num = int(target.size()[0])
    kernels = gaussian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:source_num, :source_num].mean()
    YY = kernels[source_num:, source_num:].mean()
    XY = kernels[:source_num, source_num:].mean()
    YX = kernels[source_num:, :source_num].mean()
    # print(XX.shape, YY.shape, XY.shape, YX.shape)
    # loss = torch.mean(XX + YY - XY -YX)
    loss = XX + YY - XY - YX

    return loss#因为一般都是n==m，所以L矩阵一般不加入计算


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

    mat = scipy.io.loadmat('Syndata.mat')
    data = mat['data']
    data.shape

    K = rbf_kernel(data, data, 0.5/1.8**2)
    print(f"K: {K.shape}")
    print(K[:10])

    intervel = 10
    
    indices = []
    print(f'Selecting samples......')
    for i in range(intervel, len(data)+intervel, intervel):
        idx = halving(K, i)
        indices.append(idx)

    print(f'Done......')

    data_1 = []
    # data_2 = []
    for i, idx in enumerate(indices, 1):
        data_1.append(data[idx,:])
        # data_2.append(data[random.sample(range(0, len(data)), intervel*i), :])

    mmd_scores = []

    for i in range(len(data)//intervel):
        mmd_scores.append(mmd_rbf(torch.Tensor(data_1[i]), torch.Tensor(data)))

    shattering_ratio = [(1 - intervel*i/len(data)) for i in range (1,len(data)//intervel + 1)]

    plt.plot(shattering_ratio, mmd_scores)
    plt.xlabel('Shattering ratio')
    plt.ylabel('MMD loss')
    plt.show()