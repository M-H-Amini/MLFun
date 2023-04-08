import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MHDataset(Dataset):
    def __init__(self, linear_slope=0.5, sine_amp=1, sine_freq=1, n=20, x_range=(-3, 3), noise_amp=0.1, offset=1):
        self.linear_slope = linear_slope
        self.sine_amp = sine_amp
        self.sine_freq = sine_freq
        self.n = n
        self.x_range = x_range
        self.noise_amp = noise_amp
        self.offset = offset
        ##  Make data
        self.X = np.linspace(*x_range, num=n)
        self.y = self.X * linear_slope + self.sine_amp * np.sin(self.sine_freq * self.X) + self.offset
        self.y = self.y + np.random.randn(*self.y.shape)  ##  Corrupting with noise...

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.X[index:index+1], self.y[index]

if __name__ == '__main__':
    mhd = MHDataset(n=128)
    data_loader = DataLoader(mhd, batch_size=32, shuffle=False)
    X, y = next(iter(data_loader))
    plt.plot(X, y, 'r--o')
    plt.show()
    
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")



