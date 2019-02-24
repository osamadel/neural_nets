# Logistic regression
#%%
# Importing libraries
import numpy as np
import random
from matplotlib import pyplot as plt

#%%
class LogisticRegression():
    def __init__(self):
        self.w = np.random.rand(1,1)
        self.b = np.random.rand(1,1)
        self.J = 0
    
    def __sigmoid(self, z):
        return 1 / (1 + (np.exp(-1 * z)))
    
    def __step(self, w, b, X, y, alpha, m, print_params=False):
        """ Implements one step of backpropagation """
        if print_params:
            print("w.shape:",w.shape)
            print("b.shape:", b.shape)
            print("X.shape:", X.shape)
            print("y.shape:", y.shape)
        Z = np.dot(w, X) + b
        # print("Z.shape:", Z.shape)
        A = self.__sigmoid(Z)
        # print("A.shape:", A.shape)
        dZ = A - y
        # print("dZ.shape:", dZ.shape)
        dw = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        w = w - alpha * dw
        b = b - alpha * db
        J = np.sum(-y * np.log(A) - (1-y) * np.log(1-A)) / m
        return w, b, J
    
    def train(self, X, y, print_params=False):
        """ Trains logsitic regression model - X is N x M matrix and y is 1 x M matrix """
        alpha = 0.01
        m = 1000
        for i in range(100000):
            self.w, self.b, self.J = self.__step(self.w, self.b, X, y, alpha, m)
            if print_params:
                print("J:", self.J)
                print("w:", self.w, "b:", self.b)
                print("========")
        
    def predict(self, x_tst):
        """ Returns the prediction of the input x_tst - x_tst should be 1 x N vector """
        return self.__sigmoid(np.dot(self.w, x_tst) + self.b)

#%%
# Generating training data
x = np.linspace(1, 10, 1000)[:, np.newaxis].T
y = np.hstack((np.zeros((1,500)), np.ones((1,500))))

# Adding some randomness
y[0,random.randint(0,500)] = 1
y[0,random.randint(500,1000)] = 0
print("X is", x.shape)
print("y is", y.shape)

#%%
# Creating and training the model
model = LogisticRegression()
model.train(x, y)

#%%
# Generating a test example
x1 = np.array([[0.05]])
# Predicting the output
y1 = model.predict(x1)
print("Prediction of", x1, " is", y1)

#%%
# Generating predictions for all input examples X
x_test = np.linspace(1, 10, 100)[:,np.newaxis].T
y_p = model.predict(x_test)

#%%
# Plotting training data
plt.figure(0)
plt.plot(x, y, "r+")
# plt.show()
# Plotting predictions
print("x is", x.shape, "and y_p is", y_p.shape)
print("max of y_p:", np.max(y_p))
plt.plot(x_test, y_p, "bo")
plt.show()