# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 04:14:45 2023

@author: ODD
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
x=torch.randn(5,1)
y=x.pow(4)+x.pow(5)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.first_layer = nn.Linear(5, 10)
        self.final_layer = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X_batch):
        layer_out = self.relu(self.first_layer(X_batch))

        return self.softmax(self.final_layer(layer_out))
model=Classifier()
nll_loss = nn.NLLLoss()
#optimizer=torch.optim.SGD(model.parameters,lr=0.01)
losses=[]
for epoch in range(5000):
    pred_y=model(x)
    loss=nll_loss(pred_y,y)
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    #optimizer.step()
plt.plot(losses)
plt.show()