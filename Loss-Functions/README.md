# Loss functions

- Cross-entropy loss, or log loss
- Hinge - Used for classification.
```python
def Hinge(yHat, y):
    return np.max(0, 1 - yHat * y)
```

- Huber - Typically used for regression. It’s less sensitive to outliers than the MSE.
- Kullback-Leibler
- L1
```python
def L1(yHat, y):
    return np.sum(np.absolute(yHat - y))
```

- L2
```python
def L2(yHat, y):
    return np.sum((yHat - y)**2)
```

- Maximum Likelihood
- Mean Squared Error
```python
def MSE(yHat, y):
    return np.sum((yHat - y)**2) / y.size
```

- Logistic loss
- 0-1 loss function
