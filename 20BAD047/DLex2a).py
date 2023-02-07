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
class Equ(nn.Module):
    def __init__(self):
        super(Equ, self).__init__()
        self.first_layer = nn.Linear(n_input, n_hidden)
        self.final_layer = nn.Linear(n_hidden,n_out)
        self.relu = nn.ReLU()
    def forward(self, X_batch):
        layer_out = self.relu(self.first_layer(X_batch))
        return self.final_layer(layer_out)
model= Equ()
def loss_function(y,y_pred):
        return ((y_pred-y)**2).mean()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
losses = []
for epoch in range(5000):
    pred_y = model(x)
    loss = loss_function(pred_y, y)
    losses.append(loss.item())

    model.zero_grad()

    optimizer.step()
import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()
