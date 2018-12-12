# Support vector machines

## Linear SVM Regression: Primal Formula
Suppose we have a set of training data where <img src="http://latex.codecogs.com/gif.latex?x_n" border="0"/> is a multivariate set of N observations with observed response values <img src="http://latex.codecogs.com/gif.latex?y_n" border="0"/>.


To find the linear function


<img src="http://latex.codecogs.com/gif.latex?f(x)=x'\beta+b" border="0"/>


and ensure that it is as flat as possible, find <img src="http://latex.codecogs.com/gif.latex?f(x)" border="0"/> with the minimal norm value <img src="http://latex.codecogs.com/gif.latex?f(x)=(\beta'\beta)" border="0"/>. This is formulated as a convex optimization problem to minimize


<img src="http://latex.codecogs.com/gif.latex?J(\beta)=\frac{1}{2}\beta'\beta" border="0"/>


subject to all residuals having a value less than <img src="http://latex.codecogs.com/gif.latex?\epsilon" border="0"/>; or, in equation form:


<img src="http://latex.codecogs.com/gif.latex?\forall n: |y_n-(x_n'\beta+b)|\leq \epsilon." border="0"/>



It is possible that no such function <img src="http://latex.codecogs.com/gif.latex?f(x)" border="0"/> exists to satisfy these constraints for all points. To deal with otherwise infeasible constraints, introduce slack variables <img src="http://latex.codecogs.com/gif.latex?\xi_n" border="0"/> and <img src="http://latex.codecogs.com/gif.latex?\xi_n*" border="0"/> for each point.



Including slack variables leads to the objective function, also known as the primal formula:


<img src="http://latex.codecogs.com/gif.latex?J(\beta)=\frac{1}{2}\beta'\beta + C \sum_{n=1}^N (\xi_n+\xi_n^{*})," border="0"/>



subject to:







## R with `e1071` library
```R
library(e1071)

#Fit a model. The function syntax is very similar to lm function
model_svm <- svm(y ~ x , data = train, scale = TRUE, kernel ="radial")
#Use the predictions on the data
pred <- predict(model_svm, train)


svm_tune <- tune(svm, y ~ x, data = train, ranges = list(epsilon = seq(0,1,0.01), cost = 2^(2:9)))
print(svm_tune)

best_mod <- svm_tune$best.model
best_mod_pred <- predict(best_mod, train)

```
