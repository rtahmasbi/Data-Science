# Support vector machines

<img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " />



R with `e1071` library
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
