# Neural network models

Simulating data:
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
```

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
```

## Multi-layer Perceptron
Both `MLPRegressor` and `MLPClassifier` use parameter `alpha` for regularization (L2 regularization) term
which helps in avoiding overfitting by penalizing weights with large magnitudes.

Multi-layer Perceptron (MLP) is sensitive to feature scaling, so it is highly recommended to scale your data.

Python3 with `scikit-learn`
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)
```

### Classification

MLP trains using Backpropagation algorithm (Backpropagation is a method to calculate a gradient. Backpropagation is commonly used by the gradient descent optimization algorithm to adjust the weight of neurons by calculating the gradient of the loss function. Backpropagation can be used with any gradient-based optimizer, such as L-BFGS or truncated Newton).

Python3 with `scikit-learn`
```python
import numpy as np
from sklearn.neural_network import MLPClassifier

data = np.loadtxt('../Data/boston2.txt')
data.shape
X = data[:,0:12]
y = data[:,12]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)
clf.predict([[2., 2.], [-1., -2.]])
[coef.shape for coef in clf.coefs_]
```


### Regression
MLP trains using Stochastic Gradient Descent, Adam, or L-BFGS.

Python3 with `scikit-learn`
```python
import numpy as np
from sklearn.neural_network import MLPRegressor

```



R with `neuralnet` library
```R
library(neuralnet)
Boston.scaled$medv <- scale(Boston$medv, center = min.medv, scale = max.medv - min.medv)

# Train-test split
Boston.train.scaled <- Boston.scaled[Boston.split, ]
Boston.test.scaled <- Boston.scaled[!Boston.split, ]

# neuralnet doesn't accept resp~. (dot) notation
# so a utility function to create a verbose formula is used
Boston.nn.fmla <- generate.full.fmla("medv", Boston)

# 2 models, one with 2 layers of 5 and 3
# the second with one layer of 8
# linear output is used for a regression problem
Boston.nn.5.3 <- neuralnet(y, data=Boston.train.scaled, hidden=c(5,3), linear.output=TRUE)
plot(Boston.nn.5.3)

Boston.nn.8 <- neuralnet(y, data=Boston.train.scaled, hidden=8, linear.output=TRUE)

# Predicting
pr.nn <- compute(Boston.nn.5.3,test_[,1:13])

```




# Convolutional neural network (CNN)
convolutional neural network (CNN, or ConvNet) is a class of deep, feed-forward artificial neural networks.
CNNs use a variation of multilayer perceptrons designed to require minimal preprocessing. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics.


# Recurrent Neural Networks
Time series, Morkov chains
## LSTM Networks
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
python with TensorFlow:
```python
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
```
