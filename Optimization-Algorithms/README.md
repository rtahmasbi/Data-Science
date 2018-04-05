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
