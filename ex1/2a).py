# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:30:52 2023

@author: vishn
"""
import torch  
import torch.nn as nn  
import matplotlib.pyplot as plt  
import numpy as np  
#x=torch.tensor([1,2,3,4,5]) 
#y=x.pow(4)+x.pow(5)  
n_input, n_hidden, n_out, batch_size, learning_rate = 1, 1, 1, 5, 0.01
x = torch.randn(batch_size, n_input)
y = x.pow(4)+x.pow(5)
print(x)
print(y)
print(x.size())
print(y.size())
model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())
print(model)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
losses = []
for epoch in range(5000):
    pred_y = model(x)
    loss = loss_function(pred_y, y)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()
import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()
