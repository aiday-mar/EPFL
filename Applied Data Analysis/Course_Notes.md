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

The MapReduce algorithm schedules tasks, does a virtualization of the file system, there is a fault tolerance, and does job monitoring. Spark is a high level API for programming MapReduce like jobs. An example is :

```
sc = SparkContext()
print “I am a regular Python program, using the pyspark lib”
users = sc.textFile(‘users.tsv’)  # user <TAB> age
          .map(lambda s: tuple(s.split(‘\t’)))
          .filter(lambda (user, age): age>=18 and age<=25)
pages = sc.textFile(‘pageviews.tsv’)  # user <TAB> url
          .map(lambda s: tuple(s.split(‘\t’)))
counts = users.join(pages)
              .map(lambda (user, (age, url)): (url, 1)
              .reduceByKey(add)
              .takeOrdered(5)
```

Next we study RDDs which are resilient distributed datasets. The python script is run on the driver, and the RDD operatios are run on executors. Next we can operate transformations or actions on RDD. We have different RDD transformations, such as the following. The `map(func)` returns a new distributed dataset formed by passing each element of the source through a function func. The method `filter(func)` returns a new dataset formed by selecting those elements of the source on which func returns true. Next we have `flatMap(func)`, which is similar to the map function but each inputitem can be mapped to 0 or more output items. The `sample(withReplacement?, fraction, seed)` method samples a fraction fraction of the data, with or without replacement, using a given random number generator seed. The `union(otherDataset)` returns a new dataset that contains the union of the elements in the source dataset and the argument. Same for `intersection(otherDataset)`. We also have the `distinct()` methods which returns a new dataset that contains the distinct elements of the source dataset.

We also consider the following methods. The method `groupByKey()` is called on a dataset of (K, V) pairs and returns a dataset of (K, Iterable<V>) pairs. We have for example : `{(1,a), (2,b), (1,c)}.groupByKey() → {(1,[a,c]), (2,[b])}`. We also have the following method `reduceByKey(func)` which when called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs where the values for each key are aggregated using the given reduce function func, which must be of type (V, V) => V. For example `{(1, 3.1), (2, 2.1), (1, 1.3)}.reduceByKey(lambda (x,y): x+y) -> {(1, 4.4), (2, 2.1)}`.
