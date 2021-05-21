import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
import scipy.io
import matplotlib.pyplot as plt
from halving import halving
from mmd_loss import mmd_rbf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--interval',
                    type=int,
                    default=10,
                    help='increase of samples')

opt = parser.parse_args()
print(opt)

interval = opt.interval

mat = scipy.io.loadmat('Syndata.mat')
data = mat['data']
print(f'Data: {data.shape}')

K = rbf_kernel(data, data, 0.5/1.8**2)
print(f"K: {K.shape}")

print(f'Selecting samples......')

for num_samples in range(interval, len(data)+interval, interval):
    
    idx = halving(K, num_samples)
    mmd_loss = mmd_rbf(torch.Tensor(data[idx,:]), torch.Tensor(data))
    
    if mmd_loss < 1.5e-2:
        break

    shattering_ratio = 1 - num_samples/len(data)

print(f'Done......')

print(f'Shattering ratio: {shattering_ratio}\nMMD loss: {mmd_loss}')