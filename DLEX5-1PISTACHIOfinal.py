#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
n_epochs = 3
batch_size_test = 3000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# In[2]:


import torch
import torch.nn as nn
import torchvision


# In[3]:


class RBFNet(nn.Module):
    def __init__(self):
        super(RBFNet, self).__init__()
        self.hidden_dim = 100
        self.output_dim = 4
        self.num_centers = 4
        
        self.centers = nn.Parameter(torch.randn(4, 12288))
        self.beta = nn.Parameter(torch.ones(num_centers, 1) / num_centers)
        self.sigma = sigma
        
        self.fc = nn.Linear(num_centers, output_dim)
    
    def radial_basis(self, x):
        C = self.centers.view(self.num_centers, -1)
        return torch.exp(-torch.sum((x - C) ** 2, dim=1) / (2 * self.sigma ** 2))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        H = self.radial_basis(x)
        out = self.fc(H)
        return out


# In[4]:


hidden_dim = 100
output_dim = 4
num_centers= 4
sigma = 1.0


# In[5]:


import torchvision.transforms as transforms
import torchvision.datasets as datasets
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('C:/Users/dllab.DLL-040/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset', transform=train_transforms)
test_dataset = datasets.ImageFolder('C:/Users/dllab.DLL-040/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset/', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)


# In[6]:


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape
#shape12288


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(4):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])


# In[6]:


criterion = nn.CrossEntropyLoss()
# specify optimizer
model=RBFNet()
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('pistachio')
running_loss = 0.0
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs
for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for i, datas in enumerate(train_loader,0):
        data, target = datas
        # clear the gradients of all optimized variables
        #optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        #model=RBF(data)
        model.train()
        output = model(data)
        # calculate the loss
        loss = criterion(output.float(), target.float())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        #optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,train_loss))
    running_loss += loss.item()
    if i % 1000 == 999:    
        writer.add_scalar('training loss',running_loss / 1000,epoch * len(trainloader) + i)
        writer.add_figure('predictions vs. actuals',plot_classes_preds(net, inputs, labels),global_step=epoch * len(trainloader) + i)
        running_loss = 0.0
print('Finished Training')


# In[7]:


import torch
import torch.nn as nn
import tensorboard
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('C:/Users/dllab.DLL-040/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset', transform=train_transforms)
test_dataset = datasets.ImageFolder('C:/Users/dllab.DLL-040/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/pistachio')
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
writer.add_image('pistachio', img_grid)
writer.close()


# In[11]:


get_ipython().run_line_magic('reload_ext', 'tensorboard')


# In[13]:


get_ipython().system('pip install tensorboard')


# In[12]:


get_ipython().run_line_magic('tensorboard', '--logdir=runs')


# In[ ]:




