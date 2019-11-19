import numpy as np
import torch

def obj_load(filename, device):
    V, T, Vi, Ti = [], [], [], []
    with open(filename) as f:
       for line in f.readlines():
           if line.startswith('#'): continue
           values = line.split()
           if not values: continue
           if values[0] == 'v':
               V.append([float(x) for x in values[1:4]])
           elif values[0] == 'vt':
               T.append([float(x) for x in values[1:3]])
           elif values[0] == 'f' :
               Vi.append([int(indices.split('/')[0]) for indices in values[1:]])
               Ti.append([int(indices.split('/')[1]) for indices in values[1:]])
    return torch.Tensor(np.array(V)).to(device), \
        torch.Tensor(np.array(T)).to(device), \
        torch.Tensor(np.array(Vi)-1).to(device).long(), \
        torch.Tensor(np.array(Ti)-1).to(device).long()
