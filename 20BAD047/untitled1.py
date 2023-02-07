# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 04:52:41 2023

@author: ODD
"""
import pandas as pd
import torch
X = torch.tensor([[1,1],[1,0],[0,1],[0,0]], dtype=torch.float)
y = torch.tensor([0,1,1,1], dtype=torch.float)
print(X.shape, y.shape)
samples, features = X.shape
from torch import nn
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.first_layer = nn.Linear(features, 5)
        self.second_layer = nn.Linear(5, 10)
        self.final_layer = nn.Linear(10,1)
        self.relu = nn.ReLU()

    def forward(self, X_batch):
        layer_out = self.relu(self.first_layer(X_batch))
        layer_out = self.relu(self.second_layer(layer_out))

        return self.final_layer(layer_out)
regressor = Regressor()
preds = regressor(X[:4])
print(preds)