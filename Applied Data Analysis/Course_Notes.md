# Course Notes for Applied Data Analysis

**Starting from Lecture 6**

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

