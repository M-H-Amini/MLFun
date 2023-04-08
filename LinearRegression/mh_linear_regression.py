import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mh_data import MHDataset

class MHLinearRegression(Module):
    def __init__(self, input_dim:int):
        self.W = torch.randn(input_dim, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def forward(self, x):
        z = torch.matmul(x, self.W)
        z = z + self.b 
        return z
        
def trainEpoch(train_loader, model, opt, lossFunc, epoch):
    losses = []
    for (X, y) in train_loader:
        z = model.forward(X.float())
        loss = lossFunc(z, y.float())
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()    
    
    print(f'Training epoch {epoch}...Train Loss: {np.mean(losses)}')

def evalEpoch(test_loader, model, lossFunc, epoch):
    losses = []
    with torch.no_grad():
        for (X, y) in test_loader:
            z = model.forward(X.float())
            losses.append(lossFunc(z, y.float()).item())
    
    print(f'Validating epoch {epoch}...Validation Loss: {np.mean(losses)}')
        

def visualize(loader, model, title=''):
    X_eval, y_eval, pred_eval = [], [], []
    for (X, y) in loader:
        X_eval.append(X)
        y_eval.append(y)
        pred_eval.append(model.forward(X.float()))

    X_eval = torch.concat(X_eval).detach()
    y_eval = torch.concat(y_eval).detach()
    pred_eval = torch.concat(pred_eval).detach()
    
    plt.plot(X_eval, y_eval, 'rx')
    plt.plot(X_eval, pred_eval, 'b--')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    ##  Dataset Preparation...
    train_dataset = MHDataset(1, 0.1, 0.1, 128, x_range = (-3, 3), noise_amp=0.01)
    test_dataset = MHDataset(1, 0.1, 0.1, 128, x_range = (3, 6), noise_amp=0.01)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    ##  Linear Regression Model...
    model = MHLinearRegression(1)

    ##  Initial Evaluation...
    visualize(train_loader, model, 'Performance on Training Set')
    visualize(test_loader, model, 'Performance on Test Set')

    ##  Optimizer...
    opt = torch.optim.SGD([model.W, model.b], lr = 0.01)
    MSELoss = torch.nn.MSELoss()

    ##  Train...
    for i in range(100):
        trainEpoch(train_loader, model, opt, MSELoss, i+1)
        evalEpoch(test_loader, model, MSELoss, i+1)
    
    ##  Final Evaluation...
    visualize(train_loader, model, 'Performance on Training Set')
    visualize(test_loader, model, 'Performance on Test Set')

