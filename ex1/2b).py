# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:27:53 2023

@author: vishn
"""

import torch  
import torch.nn as nn  
import matplotlib.pyplot as plt  
import numpy as np  
import torch.nn.functional as F
#x=torch.tensor([1,2,3,4,5]) 
#y=x.pow(4)+x.pow(5)  
x=torch.tensor([[1.,1.],[1.,0.],[0.,1.],[0.,0.]],requires_grad=True).float()
#x = torch.randn(4, 2, requires_grad=True).float()
#x=(torch.rand(size=(4, 2)) < 0.5).float()
y = torch.tensor([0,1,1,0])
print(x)
print(y)
print(x.size())
print(y.size())
model = nn.Sequential(nn.Linear(4, 1),
                      nn.Linear(1, 1),
                      nn.Sigmoid())
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
output=F.nll_loss(F.log_softmax(x, dim=1), y)
losses=[]
for epoch in range(5000):
    pred_y = model(x)
    loss = output(pred_y, y)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()
import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(0.01))
plt.show()
