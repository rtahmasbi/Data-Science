# Nearest Neighbors

R with `class` library
```R
library(class)

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
knn(train, test, cl, k = 3, prob=TRUE)
#knn(train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)
attributes(.Last.value)

```
