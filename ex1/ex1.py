import torch
import pandas as pd
import matplotlib.pyplot as plt
ae= pd.read_excel("C:/Users/vishn/OneDrive/Desktop/Mepco/5th sem/ML/Acoustic_Extinguisher_Fire_Dataset.xlsx")
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
ae['FUEL']= label_encoder.fit_transform(ae['FUEL'])
X =torch.tensor(ae.drop("STATUS", axis=1).values, dtype=torch.float)
y = torch.tensor(
    [0 if st == 0 else 1 if st ==1 else 2 for st in ae["STATUS"]], 
    dtype=torch.long
)
print(X.shape, y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_train, X_test, Y_train, Y_test = torch.tensor(X_train, dtype=torch.float32),torch.tensor(X_test, dtype=torch.float32),torch.tensor(Y_train, dtype=torch.float32),torch.tensor(Y_test, dtype=torch.float32)
samples, features = X_train.shape
from torch import nn
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.first_layer = nn.Linear(features, 30)
        self.second_layer = nn.Linear(30, 60,bias=True)
        self.third_layer = nn.Linear(60, 90)
        self.fourth_layer = nn.Linear(90, 120)
        self.fifth_layer = nn.Linear(120, 150)
        self.sixth_layer = nn.Linear(150, 180)
        self.final_layer = nn.Linear(180,1)
        self.relu = nn.ReLU()
    def forward(self, X_batch):
        layer_out = self.relu(self.first_layer(X_batch))
        layer_out = self.relu(self.second_layer(layer_out))
        layer_out = self.relu(self.third_layer(layer_out))
        layer_out = self.relu(self.fourth_layer(layer_out))
        layer_out = self.relu(self.fifth_layer(layer_out))
        layer_out = self.relu(self.sixth_layer(layer_out))
        return self.final_layer(layer_out)
regressor = Regressor()
preds = regressor(X_train[:5])
losses=[]
print(preds)
def TrainModel(model, loss_func, optimizer, X, Y, epochs=500):
    for i in range(epochs):
        preds = model(X) ## Make Predictions by forward pass through network

        loss = loss_func(preds.ravel(), Y) ## Calculate Loss
        losses.append(loss)
        if i % 100 == 0: ## Print MSE every 100 epochs
            print("MSE : {:.2f}".format(loss))
from torch.optim import SGD, RMSprop, Adam

torch.manual_seed(42) 

epochs = 500
learning_rate = torch.tensor(1/1e3) # 0.001

regressor = Regressor()
mse_loss = nn.MSELoss()
optimizer = SGD(params=regressor.parameters(), lr=learning_rate)

TrainModel(regressor, mse_loss, optimizer, X_train, Y_train, epochs=epochs)
test_preds = regressor(X_test) ## Make Predictions on test dataset

train_preds = regressor(X_train) ## Make Predictions on train dataset

from sklearn.metrics import r2_score
print(losses)
print("Train R^2 Score : {:.2f}".format(r2_score(train_preds.detach().numpy().squeeze(), Y_train.detach().numpy())))
print("Test  R^2 Score : {:.2f}".format(r2_score(test_preds.detach().numpy().squeeze(), Y_test.detach().numpy())))