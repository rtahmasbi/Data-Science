# Generalized Linear Models

## Ordinary Least Squares

R
```[R]
lm(y~X)
```

scikit-learn
```[python]
import numpy as np
from sklearn import linear_model

data = np.loadtxt('../Data/boston2.txt')
data.shape
X = data[:,0:12]
y = data[:,12]
reg = linear_model.LinearRegression()
reg.fit(X,y)
reg.coef_
```


## Ridge Regression
## Lasso
## Multi-task Lasso
## Elastic Net
## Multi-task Elastic Net
## Least Angle Regression
## LARS Lasso
## Orthogonal Matching Pursuit (OMP)
## Bayesian Regression
## Bayesian Ridge Regression
## Automatic Relevance Determination - ARD
## Logistic regression
## Stochastic Gradient Descent - SGD
## Perceptron
## Passive Aggressive Algorithms
## Huber Regression
## Polynomial regression
