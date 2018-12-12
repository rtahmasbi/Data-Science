# Generalized Linear Models



Simulating data:
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
```

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
```



## Ordinary Least Squares

R
```R
data <- read.table('../Data/boston2.txt')
dim(data)
reg <- lm(V13~., data=data)
coef(reg)
#(Intercept)           V1           V2           V3           V4           V5
#30.149911599  0.096152055  0.014499321  0.092071324 -1.015307104  5.196527182
#         V6           V7           V8           V9          V10          V11
#-4.409321343  0.088900885  0.150016054  0.048212135 -0.001145105  0.117746786
#        V12
#-0.008314356

```

Python3 with `scikit-learn`
```python
import numpy as np
from sklearn import linear_model

data = np.loadtxt('../Data/boston2.txt')
data.shape
X = data[:,0:12]
y = data[:,12]
reg = linear_model.LinearRegression()
reg.fit(X,y)
reg.intercept_
#30.149911599434258
reg.coef_
#array([ 9.61520551e-02,  1.44993215e-02,  9.20713243e-02, -1.01530710e+00,
#        5.19652718e+00, -4.40932134e+00,  8.89008851e-02,  1.50016054e-01,
#        4.82121354e-02, -1.14510513e-03,  1.17746786e-01, -8.31435585e-03])

```


Python3 with `TensorFlow`
```python
import numpy as np
import tensorflow as tf
data = np.loadtxt('../Data/boston2.txt')
data = np.matrix(data)
data.shape

XX = data[:,range(12)]
YY = data[:,12]
X = tf.placeholder(tf.float64, shape=(None,12), name="X")
Y = tf.placeholder(tf.float64, shape=(None,1), name="Y")
w = tf.Variable(np.zeros([12,1]), name="weights", dtype=tf.float64, trainable=True)
b = tf.Variable(np.mean(YY), name="bias", trainable=True, dtype=tf.float64)
Y_predicted = tf.matmul(X,w) + b
loss = tf.losses.mean_squared_error(Y , Y_predicted)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate=10).minimize(loss)
nepochs = 50000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(nepochs): # run epochs
        sess.run(optimizer,feed_dict={X:XX, Y:YY})
        ll = sess.run(loss,feed_dict={X:XX, Y:YY})
        if (i%1000==0):
            print("iteration = {i:7d}, loss: {ll:4.3f}".format(i=i, ll=ll))
    w_hat, b_hat = sess.run([w, b])
    print("b_hat = %s" %  b_hat)
    print("w_hat = %s" %  w_hat)

```



Python3 with `TensorFlow`
```python
import numpy as np
import tensorflow as tf
data = np.loadtxt('../Data/boston2.txt')
data = np.matrix(data)
data.shape

columns = tf.feature_column.numeric_column('x',shape=(12,), dtype=tf.float32)
lin_reg = tf.estimator.LinearRegressor(feature_columns=[columns])

# Train the estimator
train_input = tf.estimator.inputs.numpy_input_fn(x={"x": data[:,range(12)]}, y=data[:,12], shuffle=False, num_epochs=None)
lin_reg.train(train_input,steps=2500)

# Make two predictions
predict_input = tf.estimator.inputs.numpy_input_fn(x={"x": data[0:10,range(12)]}, num_epochs=1, shuffle=False)
results = lin_reg.predict(predict_input)

# Print result
for value in results:
    print(value['predictions'])

```





## Ridge Regression
Ridge regression uses L2 regularisation to weight/penalise residuals when the parameters of a regression model are being learned.

Ridge regression or in machine learning it is known as weight decay, or Tikhonov-Miller method, the Phillips--Twomey method, the constrained linear inversion method, is the most commonly used method of regularization of \emph{ill-posed} problems.


Suppose that for a known matrix <img src="http://latex.codecogs.com/gif.latex?A"/> and vector <img src="http://latex.codecogs.com/gif.latex?\mathbf{b}"/>, we wish to find a vector <img src="http://latex.codecogs.com/gif.latex?\mathbf{x}"/>  such that:

<img src="http://latex.codecogs.com/gif.latex?A\mathbf{x} =\mathbf{b}"/>


The standard approach is ordinary least squares linear regression. However, if no <img src="http://latex.codecogs.com/gif.latex?\mathbf{x}"/> satisfies the equation or more than one <img src="http://latex.codecogs.com/gif.latex?\mathbf{x}"/> does -- that is, the solution is not unique -- the problem is said to be ill posed. In such cases, ordinary least squares estimation leads to an overdetermined (over-fitted), or more often an underdetermined (under-fitted) system of equations.


Ordinary least squares seeks to minimize the sum of squared residuals, which can be compactly written as:


<img src="http://latex.codecogs.com/gif.latex?\|A\mathbf {x} -\mathbf {b} \|^{2}"/>



where <img src="http://latex.codecogs.com/gif.latex?\left\|\cdot \right\|^{2}"/> is the Euclidean norm. In order to give preference to a particular solution with desirable properties, a {\bf regularization} term can be included in this minimization:


<img src="http://latex.codecogs.com/gif.latex?\|A\mathbf{x}-\mathbf {b} \|^{2}+\|\Gamma\mathbf{x}\|^{2}"/>



for some suitably chosen Tikhonov matrix, <img src="http://latex.codecogs.com/gif.latex?\Gamma"/>. In many cases, this matrix is chosen as a multiple of the identity matrix (<img src="http://latex.codecogs.com/gif.latex?\Gamma=\alpha&space;I"/>), giving preference to solutions with smaller norms; this is known as <img src="http://latex.codecogs.com/gif.latex?L_2"/> regularization.


An explicit solution, denoted by <img src="http://latex.codecogs.com/gif.latex?\hat{\mathbf{x}}"/>, is given by:


<img src="http://latex.codecogs.com/gif.latex?\hat{\mathbf {x}}=(A^{\top}A+\Gamma^{\top}\Gamma)^{-1}A^{\top}\mathbf{b}"/>


The effect of regularization may be varied via the scale of matrix <img src="http://latex.codecogs.com/gif.latex?\Gamma"/>.  For <img src="http://latex.codecogs.com/gif.latex?\Gamma =0"/> this reduces to the unregularized least squares solution provided that <img src="http://latex.codecogs.com/gif.latex?{(A^\top&space;A)}^{-1}"/> exists.




### R with `glmnet` library

```R
library(glmnet)
data <- read.table('../Data/boston2.txt')
dim(data)
X <- as.matrix(data[,1:12])
y <- data[,13]

lambdas <- 10^seq(3, -2, by = -.1)

fit <- glmnet(X, y, alpha = 0, lambda = lambdas) #alpha=0 is ridge, =1 is lasso
summary(fit)
cv_fit <- cv.glmnet(X, y, alpha = 0, lambda = lambdas)
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
opt_lambda
y_predicted <- predict(fit, s = opt_lambda, newx = X)
mean((y-y_predicted)^2)

plot(y)
lines(y_predicted)
```


### R with `MASS` library
```R
library(MASS)
data <- read.table('../Data/boston2.txt')
dim(data)
X <- as.matrix(data[,1:12])
y <- data[,13]

lambdas <- 10^seq(3, -2, by = -.1)

fit <- lm.ridge(y ~ X)
summary(fit)

cv_fit <- lm.ridge(y ~ X, lambda = lambdas)
whichIsBest <- which.min(cv_fit$GCV)
opt_lambda <- cv_fit$lambda[whichIsBest]
opt_lambda


fit2 <- lm.ridge(y ~ X, lambda = opt_lambda)
y_predicted <- as.matrix(cbind(const=1,X)) %*% coef(fit2)
mean((y-y_predicted)^2)

plot(y)
lines(y_predicted)
```


### Python3 with `scikit-learn`
```python
import numpy as np
from sklearn import linear_model
from sklearn import metrics

data = np.loadtxt('../Data/boston2.txt')
data.shape
X = data[:,0:12]
y = data[:,12]

reg = linear_model.Ridge(alpha = .5)
reg.fit(X,y)

y_predict = reg.predict(X)

mse = metrics.mean_squared_error(y, y_predict)
mse
```


## Lasso

R with `glmnet` library
```R
library(glmnet)

lambda <- 10^seq(10, -2, length = 100)
lasso.mod <- glmnet(x[train,], y[train], alpha = 1, lambda = lambda) #alpha=0 is ridge, =1 is lasso
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test,])

```

Python3 with `scikit-learn`
```python
from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])
```

## Multi-task Lasso
## Elastic Net
Python3 with `scikit-learn`
```python
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(X, y)
regr.intercept_
regr.coef_
regr.predict([[0, 0]])
```

## Multi-task Elastic Net

Python3 with `scikit-learn`
```python
from sklearn import linear_model
clf = linear_model.MultiTaskElasticNet(alpha=0.1)
clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
```


## Least Angle Regression
## LARS Lasso
## Orthogonal Matching Pursuit (OMP)
## Bayesian Regression
## Bayesian Ridge Regression
## Automatic Relevance Determination - ARD
## Logistic regression

R
```R
data <- read.table('../Data/boston2.txt')
dim(data)
X <- as.data.frame(data[,1:12])
y <- data[,13]
y_bin <- (y>mean(y)) * 1 # create binary var
model <- glm(y_bin ~ X,family=binomial(link='logit'))
summary(model)

y_predicted <- predict(model,newdata=X,type='response')

y_predicted <- predict(fit, s = opt_lambda, newx = X)
mean((y-y_predicted)^2)

plot(y)
lines(y_predicted)

```

## Stochastic Gradient Descent - SGD
Python3 with `scikit-learn`
```python
from sklearn import linear_mode
clf = linear_model.SGDRegressor()
clf.fit(X, Y)
clf.predict([[-0.8, -1]])
```


## Perceptron
## Passive Aggressive Algorithms
## Huber Regression
## Polynomial regression
