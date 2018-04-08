# Optimization algorithms

1. Gradient descent (steepest descent) -> gradient vector, and hence it is a first order method
2. Newton's method -> second order algorithm because it makes use of the Hessian matrix. The training rate, \etha, can either be set to a fixed value or found by line minimization.
3. Conjugate gradient -> something intermediate between gradient descent and Newton's method. Here \gamma is called the conjugate parameter, and there are different ways to calculate it. Two of the most used are due to Fletcher and Reeves and to Polak and Ribiere. For all conjugate gradient algorithms, the training direction is periodically reset to the negative of the gradient.
4. Quasi-Newton method -> approximate the inverse Hessian by another matrix G, using only the first partial derivatives of the loss function. Two of the most used are the Davidon–Fletcher–Powell formula (DFP) and the Broyden–Fletcher–Goldfarb–Shanno formula (BFGS).
5. Levenberg-Marquardt algorithm (damped least-squares method) -> has been designed to work specifically with loss functions which take the form of a sum of squared errors. It works with the gradient vector and the Jacobian matrix.

```
Memory
^
|
|            Levenberg Marquardt
|         Quasi Newton
|      Newton
|   Conjugate gradient
| GD
--------------------------> Speed
```



## TensorFlow optimizers
- `tf.train.GradientDescentOptimizer` gradient descent algorithm
- `tf.train.AdadeltaOptimizer`  per-dimension learning rate method- The method dynamically adapts over time using only first order information
- `tf.train.AdagradOptimizer` The algorithms dynamically incorporate knowledge of the geometry of the data observed in earlier iterations to perform more informative gradient-based learning. Informally, The procedures give frequently occurring features very low learning rates and infrequent features high learning rates, where the intuition is that each time an infrequent feature is seen, the learner should “take notice.”
- `tf.train.AdagradDAOptimizer`
- `tf.train.MomentumOptimizer` Momentum is a method that helps accelerate Stochastic Gradient Descent (SGD) in the relevant direction and dampens oscillations
- `tf.train.AdamOptimizer` Adaptive Moment Estimation (Adam). Adam is similar to SGD in a sense that it is a stochastic optimizer, but it can automatically adjust the amount to update parameters based on adaptive estimates of lower-order moments.
- `tf.train.FtrlOptimizer` Follow the Regularized Leader
- `tf.train.ProximalGradientDescentOptimizer`
- `tf.train.ProximalAdagradOptimizer`
- `tf.train.RMSPropOptimizer`
