# Course Notes for Applied Data Analysis

**Lecture 6**

In supervised learning you want to find the function f approximating the values y using the input X. In unsupervised machine learning you transform the input essentially into an output y.

The k Nearest Neighbord problem can be interepreted in different ways, for one you may want to use a calssification technique and find the k nearest outputs. You can also decide to use a regression technique and suppose y is the average value of the k nearest neighbors. The distance can be measured in different ways, using the euclidean metric. We can call the input X as the features and the output y as the label. The bias essentially calculates the expected difference between the image by the statical estimate of the input and the actual input. The total expected error is the bias squared added to the variance. 

The curse of dimensionality occurs when you have a lot of dimensions and few points actually close to the point you are currently considering. However data generally lives in close clusters in low dimensions. In the decision tree you have a question at the node, the branches are various answers and leaves are final classes. You start by partitioning data from the root. You use the tree to find which people give positive/negative answers. From this we can infer the amount of entropy from the set S. This entrop is given by :

```
H(P,N) = - P/(P+N) log(P/(P+N)) - N/(P+N)log(N/(P+N))
```

Suppose that we have an attribute A partitioned into v partitions S, meaning S1, S2 ... Sv. Now the entropy of attribute A is :

```
H(A) = sum_i=+^n (Pi + Ni)/(P + N) H(Pi, Ni)
```

The information gain obtained by splitting S using A is : `Gain(A) = H(P, N) - H(A)`.

In general, Decision Tree algorithms are referred to as CART or Classification and Regression Trees. So, what is actually going on in the background? Growing a tree involves deciding on which features to choose and what conditions to use for splitting, along with knowing when to stop. This algorithm is recursive in nature as the groups formed can be sub-divided using same strategy. Due to this procedure, this algorithm is also known as the greedy algorithm, as we have an excessive desire of lowering the cost. This makes the root node as best predictor/classifier.

Ensemble methods are like crowsourced machine learning algorithms. In bagging you use the algorithms on different samples of data and then combineby voting for which algorithm is the best or by averaging the outputs. In stacking you combine model outputs and for this you can use linear regression. In boosting you train the learner but after filtering, weighting the samples based on the output of previous train/test runs.

Random forests are created as follows : you grow K trees on datasets sampled from the original dataset. The subdata sets are of size N. Then you select m out of p features at each node, and then you aggregate the predictions at each tree. Boosted decision trees are trained sequentially by boosting : each tree is trained to predict error residuals of previous trees.

In the logistic regression we solve `beta^T X = log(y/(1-y))`, which gives us : `y = 1/(1 + exp(-beta^T X))`. In linear regression we have `y = X . beta`. Where we have X has the input samples and the rows of X contains the distinct observations, while the columns of X are the input features.

We have the following R-squared value, which is :

```
R^2 = 1 - sum( yi - y'i)^2 / sum( yi - y' )^2
```

Where `y'` is the sample mean and `y'i` are the components of the vector of the predicted values.

**Lecture 7**

Deep learning is used to learn features and models together, and reinforcing each other. In some classifiers you will need to take your feature and place into discrete categories. In this discretization, we could have an equal width separation within the range, or it can be an equal frequency separation meaning that in each separation there is an equal number of values or there is clustering. In supervised discretization you test the hypothesis that the membership in two adjacent intervals of a feature is independent of a class. If the memberships are independent, then the intervals should be merged. In the rankinng of the features, there are ways to measure the features. With the correlation coefficient for example, or using other possible measures. 

The X^2 method is a test on if the feature is independent of the label. Or in the forward (backward) selection, you add or remove the features and in such a way evaluate the datasets. 

You can scale the values taken by the features as follows, by performing the following transformation : xi' = ( xi - mi )/( Mi - mi ). We measure the accuracy with the following equation where TP indicates the number of true positives and TN measures the number of true negatives. We have : A = (TP + TN)/N, where N is the number of cases. The precision measures what fraction of positive predictions are actually positive using : P = (TP)/(TP + FP). The recall measure measures what fraction of positive examples did we recognize as such. We have the measure is R = TP/(TP + FN). The F-score is the harmonic mean of the precision and the recall. We have : F = 2 PR/(P+R).

**Lecture 8**

