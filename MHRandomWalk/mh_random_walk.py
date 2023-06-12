import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MHRandomWalk:
    def __init__(self, p=0.5, N=10):
        self.p = p
        self.N = N
        self.x = np.zeros(N+1, dtype=int)
        self.s = np.zeros(N+1, dtype=int)
        self.generate()

    def generate(self):
        for i in range(self.N):
            if np.random.rand() < self.p:
                self.x[i+1] = 1
            else:
                self.x[i+1] = - 1
        self.s = np.cumsum(self.x)

    def plot(self):
        sns.set()
        plt.figure()
        plt.plot(self.s, 'b--o')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.yticks(range(int(self.s.min()), int(self.s.max())+1))
        plt.show()