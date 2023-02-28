# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:26:01 2023

@author: vishn
"""
import torch
torch.device('cpu')
a=torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float32)
c=torch.zeros((3,3),dtype=torch.float32)
d=0
e=0
f=0
for i in range(0,3):
    for j in range(0,3):
        if(i==0 and j==0):
            d=a[i][j]*(((a[i+1][j+1])*(a[i+2][j+2]))-((a[i+1][j+2])*(a[i+2][j+1])))
            print(d)
        if(i==0 and j==1):
            e=a[i][j+1]*(((a[i+1][j])*(a[i+2][j+1]))-((a[i+1][j+1])*(a[i+2][j])))
            print(e)
        if(i==0 and j==2):
            f=a[i][j]*(((a[i+1][j-1])*(a[i+2][j]))-((a[i+1][j])*(a[i+2][j-1])))
            print(f)
det=0
det=(d-e+f)
print(det)