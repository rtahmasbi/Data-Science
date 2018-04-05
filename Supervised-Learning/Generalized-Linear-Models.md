# Generalized Linear Models

```
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
```

```
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
```


## Ordinary Least Squares

R
```[R]
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
reg.intercept_
#30.149911599434258
reg.coef_
#array([ 9.61520551e-02,  1.44993215e-02,  9.20713243e-02, -1.01530710e+00,
#        5.19652718e+00, -4.40932134e+00,  8.89008851e-02,  1.50016054e-01,
#        4.82121354e-02, -1.14510513e-03,  1.17746786e-01, -8.31435585e-03])

```


TensorFlow
```[python]

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





TensorFlow
```[python]
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

```
from sklearn import linear_model
reg = linear_model.Ridge(alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.coef_
```


## Lasso

from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])

## Multi-task Lasso
## Elastic Net
from sklearn.linear_model import ElasticNet
regr = ElasticNet(random_state=0)
regr.fit(X, y)
regr.intercept_
regr.coef_
regr.predict([[0, 0]])


## Multi-task Elastic Net

from sklearn import linear_model
clf = linear_model.MultiTaskElasticNet(alpha=0.1)
clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])


## Least Angle Regression
## LARS Lasso
## Orthogonal Matching Pursuit (OMP)
## Bayesian Regression
## Bayesian Ridge Regression
## Automatic Relevance Determination - ARD
## Logistic regression
## Stochastic Gradient Descent - SGD

from sklearn import linear_mode
clf = linear_model.SGDRegressor()
clf.fit(X, Y)
clf.predict([[-0.8, -1]])


## Perceptron
## Passive Aggressive Algorithms
## Huber Regression
## Polynomial regression
