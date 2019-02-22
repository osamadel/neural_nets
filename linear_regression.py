#%%
# Linear Regression Using Python

#%%
import numpy as np
import random
from matplotlib import pyplot as plt

#%%
# Let's assume we have a straight line:
# y = 4 * x + 10
# We need to create a dataset that approcimately
# represents this line.
x = np.random.randn(1000, 1) + 4
rand_bias = np.random.randn(1000, 1) 
# print(x.shape)
#%%
y = 4 * x + 10 + rand_bias
plt.figure(0)
plt.plot(x, y, "r*")
# plt.show()
#%%
x_train = x[:800,:]
y_train = y[:800,:]
x_test = x[800:,:]
y_test = y[800:,:]
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
#%%
# plt.figure(1)
# plt.plot(x_test, y_test, "r*")
# plt.show()
#%%
# Now we need to calculate to built a model with
# initialized parameters vector w
w = random.random()
b = random.random()
alpha = 0.001
thresh = 1.1
print("w:", w, "b:", b)
#%%
i = 0
while(True):
    print("Step #{0}".format(i+1))
    y_pred = w * x_train + b
    J_train = np.sum((y_pred - y_train)**2) / 800
    print("J_train:", J_train)
    dw = 2 * np.sum((y_pred - y_train) * x_train) / 800
    db = 2 * np.sum(y_pred - y_train) / 800
    w = w - alpha * dw
    b = b - alpha * db
    print("w:", w, "b:", b)
    print()
    i += 1
    if J_train < thresh:
        break

#%%
y_pred = w * x_test + b
J_test = np.sum((y_pred - y_test)**2) / 200
print("J_test:", J_test)
#%%
xline = np.array([[1], [6]])
yline = w * xline + b
plt.figure(0)

plt.plot(x, y, "r*")
plt.plot(xline, yline)
plt.show()