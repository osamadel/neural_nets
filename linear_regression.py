# Linear Regression
#%%
import numpy as np
import random
from matplotlib import pyplot as plt

#%%
class LinearRegression():
    def __init__(self):
        self.w = random.random()
        self.b = random.random()
    
    def train(self, X, y, print_params = False):
        """ X is M x 1 matrix and y is M x 1 matrix """
        alpha = 0.001
        thresh = 1.1
        i = 1
        m = y.shape[0]
        while(True):
            y_pred = self.w * X + self.b
            J_train = np.sum((y_pred - y)**2) / m
            dw = 2 * np.sum((y_pred - y) * X) / m
            db = 2 * np.sum(y_pred - y) / m
            self.w = self.w - alpha * dw
            self.b = self.b - alpha * db
            if print_params:
                print("Step #{0}".format(i))
                print("J_train:", J_train)
                print("w:", self.w, "b:", self.b)
                print()
            i += 1
            if J_train < thresh:
                print("=> Training is complete.")
                break

    
    def predict(self, x_test):
        """ Returns the prediction for all x_test """
        return self.w * x_test + self.b
#%%
# Generating random dataset
# Let's assume we have a straight line: y = 4 * x + 10,
# We need to create a dataset that approcimately represents this line.
x = np.random.randn(1000, 1) + 4
rand_bias = np.random.randn(1000, 1) 
y = 4 * x + 10 + rand_bias
print("x is", x.shape)
print("y is", y.shape)

#%%
# Splitting dataset into training and test sets
x_train = x[:800,:]
y_train = y[:800,:]
x_test = x[800:,:]
y_test = y[800:,:]
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
#%%
# Plot test set examples
# plt.figure(1)
# plt.plot(x_test, y_test, "r*")
# plt.show()
#%%
# Create and traing a linear regression model
model = LinearRegression()
model.train(x_train, y_train)

#%%
# Generating predictions for test set
y_pred = model.predict(x_test)
J_test = np.sum((y_pred - y_test)**2) / y_pred.shape[0]
print("J_test:", J_test)

#%%
# Plotting the model against the data examples
xline = np.array([[np.min(x_train)], [np.max(x_train)]])
yline = model.predict(xline)
plt.figure(0)

plt.plot(x, y, "r*")
plt.plot(xline, yline)
plt.show()