from torch import functional as F
from torch.nn import Module
from torch import Tensor
import torch
from typing import List, Optional, Any, Union

class Addnoise(Module):
    

    def __init__(self, k=10) -> None:
        super().__init__()
        self.k = k

    def forward(self, sample: Tensor) -> Tensor:
        if self.k <= 0:
            return sample
        r1 = 0
        # r2 = self.noise
        row = sample['row']
        r2 = row.max()
        # print(sample['row'])
        # row = (0.0001**0.5)* torch.randn(row.shape) # torch.randn(5)
        perm = torch.randperm(row.size(0))
        k = self.k
        idx = perm[:k]
        
        # print(row.shape, row.mean(), row.max())
        # print("k, idx", k, idx)
        samples = (r2 - r1) * torch.rand(idx.shape) + r1
        # print("samples", samples)
        # print("row[idx] before", row[idx])
        row[idx] = samples
        # print("row[idx] after", row[idx])
        # exit()
        # row *= (r2 - r1) * torch.rand(row.shape) + r1
        sample['row'] = row
        # print(sample['row'])
        # exit()
        return sample