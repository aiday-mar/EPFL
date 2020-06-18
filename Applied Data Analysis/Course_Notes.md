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
          
We also have the following transformations. The `sortByKey()` method which is called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs sorted by keys. The `join(otherDataset)` called on datasets of type (K, V) and (K, W), returns a dataset of (K, (V, W)) pairs with all pairs of elements for each key. Then we have `{(1,a), (2,b)}.join({(1,A), (1,X)}) → {(1, (a,A)), (1, (a,X))}`. Analogous methods include the leftOuterJoin, rightOuterJoin, fullOuterJoin.

We then have the following methods. The collect() method which returns all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data. The count() method which returns the number of elements in the dataset. The take(n) method which returns an array with the “first” n elements of the dataset. The saveAsTextFile(path) method which writes the elements of the dataset as a text file in a given directory in the local filesystem or HDFS.

An RDD is a list of rows, and the DataFrame is a table with rows and typed columns. In Spark SQL we have the following commands : `sc = SparkContext()`, `sqlContext = HiveContext(sc)`, `df = sqlContext.sql("SELECT * from table1 GROUP By id")`. We have an example of the logistic regression with MLLib : 

```
from pyspark.mllib.classification import LogisticRegressionWithSGD

trainData = sc.textFile("...").map(...)
testData = sc.textFile("...").map(...)

model = LogisticRegressionWithSGD.train(trainData)
predictions = model.predict(testData)
```

**Lecture 9**

We have supervised machine learning which takes input and output samples and we relate these with a function f which can have different types, classification and regression. The first is used when y is discrete and the second when the y variable is continuous, as in linear regression. In unsupervised machine learning, we have samples x of the data, we compute a function f which we apply onto x to get f(x) and thus this gives a simpler representation. When y is disrete we use the clustering technique. When y is continuous, then we have dimensionality reduction. The clustering problem issuch that given a set of points, we group the points into a number of clusters, such that the members of a cluster are close to each other, and members of different clusters are dissimilar. 

There is a clusrer bias because people assume a classification where there is none. In hierarchical clustering, the clusters form a tree-shaped hierarchy. In flat clustering there is no inter-cluster structure. Hard clustering assigns items to a unique cluster. When you apply the clusterig proble to documets, you represent a document by a vector where the ith entry is one if and only if the ith word appears in the document at any position.

We measure sets as vectors, where we measure the similarity by either the euclidean or the cosine distance. We measure sets as sets by using the Jaccard distance. There are different type of clustering like hierarchical and point assignments. In hierarchical clustering, we have agglomerative clustering where initially each point is a cluster and you repeatedly combine nearest clusters into one. We also have divisive clustering where you start with one cluster and you recursively split it. There are also point assignments clustering, where you maintain a set of clusters, and each point belongs to the nearest cluster. To represent the center of clusters you use the centroid. In the non-euclidean case you represent the center of many points with a clustroid, the point closest to other points. 

To determine the nearness of clusters you may find the minimum of the nitercluster distance or you can find the cohesion of their union. The cohesion can be the diameter of the merged cluster, meaning the maximum distance between points in the cluster, or it can be the average distance between points in the cluster. The implementation is such that at each step we compute the pairwise distance between all pairs of clusters, then merge.

Let us consider point assignment clustering. We have the example of the k-means algorithm. The goal of this is to assign each data point to one of the k clusters such that the total distance of points to their centroids is minimized. We locally minimize the euclidean distance from the data points to their respective centroids. We find the closest cluster centroid for each item and assign it to that cluster. Then we recompute the cluster centroid for each cluster.

In the K-means, we initially either choose a random sample of k points, or iteratively construct a random sample with good spacing across datasets. In the K-means++ system you first choose a cluster center at random from the data points. Then you iterate, for every remaining data point x, you compute the distance D(x) from x to the closest previously selecter cluster center. Choose a remaining point x randomly with probability proportionaly to D(x)^2 and make it a new cluster center.

To choose k, you iterate over all i, you compute the following value : s(i) = (b(i) - a(i))/(max{a(i), b(i)}), where b(i) is the average distance to points in the closest other cluster and where a(i) is the average distance to points in your own cluster. You do this for different k, then calculate for each S = average of s(i) over all poinzs i. Then plot S againt k  and pick the k for which S is the greates.

DBSCAN is the Density-Based spatial clustering of applications with noise. DBSCAN performs density-based clustering, and follows the shape of dense neighborhoods of points. Core points have at least minPts neighbors in a sphere of diameter ε around them. Core points can directly reach neighbors in their ε-sphere. From non-core points, no other points can be reached. Point q is density-reachable from p if there is a series of points p = p1, …, pn = q such that pi+1 is directly reachable from pi. All points not density-reachable from any other points are outliers.

Points p, q are density-connected if there is a point o such that both p and q are density-reachable from o. A cluster is a set of points which are mutually density-connected. That is, if a point is density-reachable from a cluster point, it is part of the cluster as well. In the above figure, red points are mutually density-reachable; B and C are density-connected; N is an outlier. We have :

```
DBSCAN(DB, dist, eps, minPts) {
   C = 0
   for each point P in database DB {
       if label(P) != undefined, then continue
       Neighbors N = RangeQuery(DB, dist, P, eps)
       if |N| < minPts then {
          label(P) = Noise
          continue
       }
       C = C + 1
       label(P) = C
       Seed set S = N \ {P}
       for each point Q in S {
          if label(Q) = Noise then label(Q) = C
          if label(Q) != undefined then continue
          label(Q) = C
          Neighbors N = RangeQuery(DB, dist, Q, eps)
          if |N| >= minPts then {
              S = S U N
          }
       }
   }
}
```

**Lecture 10**

In document retrieval, we can have a neighbor search as in kNN, where given the query q we find the k docs with smallest distance to query q. We have document classification where each document is labelled with a class. Here we train a supervised classifier based on labelled documents using the following methods : kNN, logistic regression, decision trees, random forests. Sentiment analysis can be studied with supervised learning (regression, classification). Topic detection can be studied with clustering (hierarchical or point-assignment). An example of the use of feature vectors is aht need to transform arbitrarily long strigns to fixed length vectors. You can use the nag of words method where you keep the multiplicity of the words and you don't take into account the order of the words. The bag of words matrix has one row in the matrix per document, one column per word in the dictionary.

Character encoding is the conversion of characters to bytes. Before the systems used where ASCII and Latin-1, now we use Unicode such as UTF-8, UTF-16, UTF-32. Tokenization maps character strings into a sequence of tokens/words. Stopwords are small words that don't carry a meaning such as 'a', 'of' etc. Casefolding is when you change all the words to lower case and use this to analyze data. Stemming is when you strip the suffixes of words and keep the common root. Alternatively you can extract n-grams when you have several tokens in a sequence you extract. We define the following : docfreq(w) which is the number of documents that contain word w, N which is the overall number of documents, idf(w) = log(N) - log(docfreq(w)), which is the inverse document frquency of the word. Now if the tf(w,d) term is the frquency of word w in document d, then we have that the TF-IDF matrix is such that the entry in row d and column w has value tf(w,d)*idf(w). 

Using this TF-IDF matrix let's revisit the different possible tasks. We have for example the document retrieval task. The way you do this is you compare the query doc q to all documents in the collection, and you rank the documents from the collection in an increasing order of distance. You may use for this distance measure the cosine metric, which for vectors q and v is defined as : <q/|q|, v/|v|>. The cosine distance is then 1 - cosine similarity. The second task is document classification. 

Consider also the regularization which for linear regression can be shown to be : 

```
sum_(i=1)^n ( yi - zi^t beta)^2 + lambda . sum (j=1)^p beta_j^2
```
Here we penalize very high or very low weights. 

To deal with the topic detection problem, to get the TF-IDF matrix you can multiple the docs x topics matrix by the word x topics matrix. Then the differences of the component wise elements, squared and summed is the latent semantic analysis. You can dins the SVD of the matrix TF-IDF and then the diagonal matrix will capiture the importance of the topics. 

For the taskof topic detectopn we can use the LDA method which is the Latent Dirichlet Allocation method. Use a generative story to generate a document of length n :

d = sample a topic distribution for the document
for i = 1 ... n
          t = sample a topic from the topic distribution d
          w = sample a word from the topic t
          add w to the bag of words fo the doc to be generated

Essentially in the LDA you input documents represented as bags of words, and the number K of topics to be found, and the output are the K topics themselves, and for each document we find a distribution over the K topics. How to quantify the closennes of two documents - you take the cosine of the rows of the TF-IDF matrix. How to quantify the closeness of two words - you find the cosine of the columns of the TF-IDF matrix. You can also define a matrix M where against therows we display the contexts and against the columns we display the words. An entry is : M[c, w] = log [P(c,w)/P(c).P(w)]. This is also called pointwise mutual information. The word2vec is a method which studies this pmi and uses the columns of the matrix B as word vectors. When you want to contextualize the word in a phrase you can use the BERT technology. 

The NLP pipeline allows tokenization, sentence splitting, part of speech tagging, named entity recognition. It can be implemented by the CoreNLP, nltk, spaCy.

**Lecture 11**

The projectin of a bipartite graph likely links the nodes in one part with other nodes in the same part that are at a distance of two away. Social networks for example are sparse and bounded by Dunbar's number. The degree distribution quantifies the probability that a node has degree k. The clustering coefficient denoted by Ci measures the number of edges between neighbors of node i divided by the actual number of potential edges between neighbors of node i. From this study we see the result of homophily, that similar people are more likely to be friends. In a similar way we see that most nodes are sparse but there are a few nodes, of popular people that are followed by many others. The farness of x is the average distance to x from other nodes. C(x) = 1/F(x) measures the closeness, and is only defined for connected graphs. Consider now the measure of the betweeness centrality, which is the fraction of all the shortest paths in the network that pass through node i. The katz cenrtality is a generalization of the betweeness centrality and also takes into account the neighbors at distance 2 or 3. This katz centrality is defined as follows :

```
C(i) = sum_{k=1}^{infinity} sum_{j=1}^N \alpha^k (A^k)_{ij}
```

Now consider the notion of centrality which says that the centrality is high if you receive many links from other nodes. 

```
xi = sum_j a_{ji} x_j / L_j
L(j) = sum_j a_{ji}
```
