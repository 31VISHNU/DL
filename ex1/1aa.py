# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:37:39 2023

@author: vishn
"""
import torch
torch.device('cpu')
a=torch.tensor([[1,3],[4,5]],dtype=torch.float32)
b=torch.tensor([[2,4],[9,6]],dtype=torch.float32)
c=torch.zeros((2,2),dtype=torch.float32)
ind=2
print(a.size)
for i in range(ind) :
    val1=0.0
    for j in range(ind):
        #print(i)
        val1=torch.dot(a[i,:],b[:,j])
        print("a:**",a[i,:])
        print("b:**",b[:,j])
        print(val1)
        c[i][j]=val1
print(c)

