# Logistic regression
#%%
import numpy as np
import random
from matplotlib import pyplot as plt
#%%
def sigmoid(z):
    return 1 / (np.exp(-1 * z))
#%%
def step(w, b, X, y, alpha, m):
    # print("w.shape:",w.shape)
    # print("b.shape:", b.shape)
    # print("X.shape:", X.shape)
    # print("y.shape:", y.shape)
    Z = np.dot(w, X.T) + b
    # print("Z.shape:", Z.shape)
    A = sigmoid(Z)
    # print("A.shape:", A.shape)
    dZ = A - y
    # print("dZ.shape:", dZ.shape)
    dw = np.dot(dZ, X) / m
    db = np.sum(dZ) / m
    w = w - alpha * dw
    b = b - alpha * db
    J = np.sum(-y * np.log(A) - (1-y) * np.log(1-A)) / m
    return w, b, J
#%%
x = np.linspace(1, 10, 1000)[:, np.newaxis]
y = np.hstack((np.zeros((1,500)), np.ones((1,500))))
y[0,random.randint(0,500)] = 1
y[0,random.randint(500,1000)] = 0
print(y.shape)

#%%
w = np.random.rand(1,1)
b = np.random.rand(1,1)
alpha = 0.01
m = 1000
for i in range(10000):
    w, b, J = step(w, b, x, y, alpha, m)
    print("J:", J)
    print("w:", w, "b:", b)
    print()
#%%
x1 = np.array([[700]])
y1 = sigmoid(w * x1 + b)
print(y1)
#%%
plt.figure(0)
plt.plot(x.T, y, "r*")
plt.show()