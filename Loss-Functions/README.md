# Loss functions

1- Cross-entropy loss, or log loss
2- Hinge - Used for classification.
```[python]
def Hinge(yHat, y):
    return np.max(0, 1 - yHat * y)
```

3- Huber - Typically used for regression. It’s less sensitive to outliers than the MSE.
4- Kullback-Leibler
5- L1
```[python]
def L1(yHat, y):
    return np.sum(np.absolute(yHat - y))
```

6- L2
```[python]
def L2(yHat, y):
    return np.sum((yHat - y)**2)
```

7- Maximum Likelihood
8- Mean Squared Error
```[python]
def MSE(yHat, y):
    return np.sum((yHat - y)**2) / y.size
```

9- Logistic loss
10- 0-1 loss function
