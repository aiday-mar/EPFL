# Notes from the Exercises 

**Tutorial 04**

You can import the different libraries under a different name using a smaller shortened form for example pd for panda. Then if you want to use the Python Spark common library PySpark you need to create a context as follows :

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext

# create the session
spark = SparkSession.builder.getOrCreate()

# create the context
sc = spark.sparkContext
```
You can then import the json file and use it with the spark framework as follows : `spark.read.json("text.txt")`. You can print the schema as `printSchema()`, you can `take()` and `show()`. When you want to for example print a statement you can use special formatting with variables as follows :

```
print("In total there are {0} operations".format(Bombing_Operations.count()))
```

Consider the following code below. Here you have the json data which you group according to a certai category, in each category you aggreggate all the data and gives it a new name and sort in the descending order where here you use the alias of the column.

```
missions_counts = Bombing_Operations.groupBy("ContryFlyingMission")\
                                    .agg(count("*").alias("MissionsCount"))\
                                    .sort(desc("MissionsCount"))
missions_counts.show()
```
However if you want to write everything in terms of SQL then you can transform the json file into a temporary table and perform a query on it using the sql function of the Spark library : 

```
Bombing_Operations.registerTempTable("Bombing_Operations")

query = """
SELECT ContryFlyingMission, count(*) as MissionsCount
FROM Bombing_Operations
GROUP BY ContryFlyingMission
ORDER BY MissionsCount DESC
"""

missions_counts = spark.sql(query)
missions_counts.show()
```
Then you can move the data to pandas as follows : 

```
missions_count_pd = missions_counts.toPandas()
missions_count_pd.head()
```
Now using the Pandas library you may now plot the data in the library as follows : 

```
pl = missions_count_pd.plot(kind="bar", 
                            x="ContryFlyingMission", y="MissionsCount", 
                            figsize=(10, 7), log=True, alpha=0.5, color="olive")
pl.set_xlabel("Country")
pl.set_ylabel("Number of Missions (Log scale)")
pl.set_title("Number of missions by contry")
```
You may apply some expressions on the data before you select it. Consider below where we had already selected the json file and then you run the `selectExpr` command on it. Then you convert the column of MissionDate to a date before giving it the same alias, and you also select the country where the mission was held. All this is in quotes which are doubel quotes and within square brackets. 

```
missions_countries = Bombing_Operations.selectExpr(["to_date(MissionDate) as MissionDate", "ContryFlyingMission"])
missions_countries
```
you can also group by several categories as seen below. These are separated with commas and with double quotes as follows. In the below we are likely adding an additional column which is called MissionsCount. The other columns are present too though. 

```
missions_by_date = missions_countries\
                    .groupBy(["MissionDate", "ContryFlyingMission"])\
                    .agg(count("*").alias("MissionsCount"))\
                    .sort(asc("MissionDate")).toPandas()
missions_by_date.head()
```
Next we have the following code, where we take our data and group by the country where the mission was made. Then the first variable is the country is located, and the missions table essentially contains two columns, one for the date of the mission and one for the number of missions done in that country on that day. Here you plot the date on the x axis and the number of missions on the y axis and each plot will represent a new country. 

You can also perform some searches on the data as follows :

```
jun_29_operations = Bombing_Operations.where("MissionDate = '1966-06-29' AND TargetCountry='NORTH VIETNAM'")
```

In the above we are taking the data and using the where keyword. In the parentheses we are placing the conditions that need to be verified to find the requested information. Then we find what countries have scheduled missions there on that day. We groupBy the CountryFlyingMission, count the number of rows in each group and create a new column from this called MissionsCount. Then we send this to the Pandas library. 

```
jun_29_operations.groupBy("ContryFlyingMission").agg(count("*").alias("MissionsCount")).toPandas()
```

You can put data in the cache just by typing `cache()` at the end. You can even save the data in a json file as follows : 

```
jun_29_operations.write.mode('overwrite').json("jun_29_operations.json")
```

We can also use the RDD. In the below we take the table and we analyze each row. Then the TakeOffLocation column of each row, contributes a value of 'one'.

```
all_locations = jun_29_operations.rdd.map(lambda row: (row.TakeoffLocation, 1))
locations_counts_rdd = all_locations.reduceByKey(lambda a, b: a+b).sortBy(lambda r: -r[1])
locations_counts_with_schema = locations_counts_rdd.map(lambda r: Row(TakeoffLocation=r[0], MissionsCount=r[1]))
locations_counts = spark.createDataFrame(locations_counts_with_schema)
locations_counts.show()
```
After this we take the key and consider two variable which indicate probably two rows having the same TakeOffLocation. Then you sum in such a way that you find for each TakeOffLocation how many missions were made with that specific location. Then you sort by the last column, the second column in descending order, which is why there is a negative sign. To create a schema then you create a new row, a name which is associated to the right column, and this is created into a data frame. 

To do a join, you use the following syntax :

```
tables_joined = tale1.join(table2, table1.property == table2.property)
tables_joined
```
To find for example the number of missions made by a specific aircraft you can write :

```
missions_joined = Bombing_Operations.join(Aircraft_Glossary, 
                                          Bombing_Operations.AirCraft == Aircraft_Glossary.AirCraft)
missions_joined
missions_aircrafts = missions_joined.select("AirCraftType")
missions_aircrafts.show(5)
missions_aircrafts.groupBy("AirCraftType").agg(count("*").alias("MissionsCount"))\
                  .sort(desc("MissionsCount"))\
                  .show()
```

You can also write this in pure SQL format as follows :

```

Bombing_Operations.registerTempTable("Bombing_Operations")
Aircraft_Glossary.registerTempTable("Aircraft_Glossary")

query = """
SELECT AirCraftType, count(*) MissionsCount
FROM Bombing_Operations bo
JOIN Aircraft_Glossary ag
ON bo.AirCraft = ag.AirCraft
GROUP BY AirCraftType
ORDER BY MissionsCount DESC
"""

spark.sql(query).show()
```

**Tutorial 05**

You can use collections in python as follows. A list in python is created using curly brackets as follows {}. Then you can also creates lists as follows `list([])`. This could also be done with collections like below. Here you essentially create a default dictionary which is initiliazed to some empty list. Then you can just acess a key, whether it exists or not and add a string to it. 

```
todo_list = collections.defaultdict(list)
todo_list['ADA'].append('Homework 2')
```

The collections framework also allows Counter which initialy initializes each key as having a count of zero. 

```
counter = collections.Counter()
counter['apples'] += 1
counter['oranges'] += 1
```

Here is a difference of how to code file openings :

```
# Good practice
with open('file', 'r') as file:
    print(file.read())

# Bad practice
file = open('file', 'r')
try:
    print(file.read())
finally:
    file.close()
```

Note: Using open(...) as ... automatically closes the file after the block finishes running. You can use the pickle library to store big objects in memory as follows : `pickle.load(file)`. You can also save data into a file path. Suppose the data we want to save is called result then we have :

```
def save_pickle(result, file_path = 'pickle'):
    with open(file_path, 'wb') as file:
        pickle.dump(result, file_path)
```

When you have a try catch statement in python, this is actually called a try, except statement. You have in this case then an example as follows :

```
def very_complex_operation():
    try:
        # some code
    except (FileNotFoundError, EOFError) as e:
        # some computation  
```

Note: Use %d for decimals/integers, %f for floats (alternatively %.xf to specify a precision of x), and %s for strings (or objects with string representations). We have an example as follows :

```
print("There are %d apples on the table." % (num_off_apples))
```

You can create a list directory of a folder and access all the files in a directory as follows :

```
from os import listdir

DATA_PATH = './data/'

def process_data(path = DATA_PATH):
   for file in listdir(path):
       do_something(path + file)

# Providing the target files through the command line
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--filename", help="Name of the file to process", type = str)
args = parser.parse_args()

print(args.filename)
````

You can also create lists using the following code which gives us three different elements in the list. We have : `results = [factorial(i) for i in range(3)]`. In the method we define we can raise errors as follows : `raise NotImplementedError`.

In the following code `flights = df[(df['day'] == day) & (df['airport'] == airport_id)]`, we select only some entries of the original data frame. You select the entries where the day column has entries equal to the day variable, and where the airport column has entries equal to the airport_id. 

A good example of a python code for data analysis has the imports at the top, some constants which have a capitalized name. Then in the method, we have a dataframe in which we have columns and we cast the type of each column as either an integer or a string. Then you have a main method which is called : `if __name__ == '__main__'`. You can use the `read_csv` method that takes two parameters. The first one is going to be the name of the file and the second one will be an optional parameter which reads `compression = COMPRESSION`. When you need to drop entries where there is no values you can call the `dropna()` method. To sort the columns you can write :

```
flights = flights.sort_values('hour')[['flight_id', 'dest_id']]
```

**Scaling Up Tutorial**

You can deploy a job in two ways. On a single node meaning that we use the resources of one single machine, and we distribute the tasks on multiple cores. In the cluster mode we use the resources of multiple machines. As such the cluster manager is connected to the worker nodes which contain an executor that deal with tasks. It is also connected to the driver program which deals with the SparkContext. Spark offers a unified stack. In spark there is Spark Core which is built on top of Mesos and Yarn. On top of Spark Core there is Spark SQL, Spark Streaming Real-Time, MLLib and GraphX. 

Spark SQL is a Spark module for structured data processing. It provides the data frames. Spark Streaming is an extension of the core Spark API which enables fault tolerant stream processing of live data streams. GraphX is used for graph-parallel computation. At a high level GraphX extends the Spark RDD by introducing a new Graph abstraction : a directed multigraph which properties attached to each vertex and edge. 

**Tutorial 06**

First we need to import the right packages and libraries as follows :

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
%matplotlib inline
```
We import the dataset :

```
data = pd.read_csv('data/Advertising.csv', index_col=0)
data.head()
```
Next we have :

```
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16, 8), grid=True)
data.plot(kind='scatter', x='radio', y='sales', ax=axs[1], grid=True)
data.plot(kind='scatter', x='newspaper', y='sales', ax=axs[2], grid=True)
```

Then we will use the LinearRegression algorithm of the Scikit-learn library. We have the followig code used to select certain features from the table called data :

```
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales
X.describe()
lin_reg = LinearRegression()  # create the model
lin_reg.fit(X, y)  # train it
```
From here on we can plot the predicted and the original values on one plot and then later calculate the mean squared error between the two measures. We have :

```
lr = LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, X, y, cv=5)

# Plot the results
# in the below there is only one subplot which has a specific size
# here we have the size of the corresponding plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
# here we define the range displayed on the graph
# we also see that the line is red and we use dashes
ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
ax.set_xlabel('Original')
ax.set_ylabel('Predicted')
plt.show()
mean_squared_error(y, predicted)
```
We next consider the logistic regression as follows :

```
logistic = LinearRegression()
logistic.fit(X, y)
predicted_train = logistic.predict(X)
mean_squared_error(y, predicted_train)
```

We can use the ridge regression as follows :

```
ridge = Ridge(alpha=6)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted_r = cross_val_predict(ridge, X, y, cv=5)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y, predicted_r, edgecolors=(0, 0, 0))
ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
ax.set_xlabel('Original')
ax.set_ylabel('Predicted')
plt.show()
mean_squared_error(y, predicted_r)
```

Now let us perform some classification using the logistic regression. We have :

```
titanic_raw = pd.read_excel('data/titanic.xls')
# drop from the first column the entries that have nothing in them
titanic = titanic_raw.dropna(axis=0, how='any')
titanic.head()

# We give the name dead to the entries in the table where the survived column takes value zero, and survived name to the etries where # the column takes value 1, these entries are all placed into a column which is either called dead or survived. 
dead = titanic[titanic['survived']==0]
survived = titanic[titanic['survived']==1]

print("Survived {0}, Dead {1}".format(len(dead), len(survived)))
```

Then we prepare the feature vectors for the training. We have :

```
titanic_features = ['sex', 'age', 'fare']
X = pd.get_dummies(titanic[titanic_features])
X.head()

# the label used for the training 
y = titanic['survived']
logistic = LogisticRegression(solver='lbfgs')

precision = cross_val_score(logistic, X, y, cv=10, scoring="precision")
recall = cross_val_score(logistic, X, y, cv=10, scoring="recall")

# Precision: avoid false positives
print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))
# Recall: avoid false negatives
print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))
```

**Homework 2 solutions**

First we need to import all the appropriate libraries as follows :

```
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```

Then we need to import the csv file as follows :

```
players = pd.read_csv('data/fifa19_players.csv')
players.head()
```

Then we can plot the data from the above csv using the pyplot command :

```
plt.xlabel('Skill')
plt.ylabel('#players')
plt.title('Distribution of Player Skills')
plt.hist(players['Overall'], bins=30)
```

You can after this find the data type of the table of data using the following command : `players['Overall'].dtype`.  By changing the last line of the above code to : `plt.hist(players['Overall'], np.arange(players['Overall'].min(), players['Overall'].max()))`, we find that the plot has little widths for each entry. Where you plot the players['Overall'] against a vector which ranges from the minimum value to the maximum value of tha table. Next we define the skills array, which contains the string names of the columns of the table that interest us. we then select these appropriate columny by writing `players = players[skills]`. You can also find the distinct types of values that exist in a column. For exanple when you write `players['Work Rate'].unique()`, then this outputs the unique elements in the column of work rates. We can map these different types to different numerical values as follows : 

```
# https://stackoverflow.com/questions/23586510/return-multiple-columns-from-apply-pandas
# We have here each word is mapped to a numerical value which is either 0, 1 or 2
rate_to_int = {'Low':0, 'Medium':1, 'High':2}
# Then we have this variable p which is used inside of the method
def extract_work_rate(p):
    # meaning that is the tyoe of Work rate column of matrix p is not a string
    if not type(p['Work Rate']) == str:
        attack = np.nan
        defense = np.nan
    else:
        # In the case when the type of the column is the correct string type
        # then you take the string and you split it according to this symbol "/"
        attack, defense = p['Work Rate'].split('/ ')
        # then you need to select the appropriate indices from the table and place into the appropriate vector
        attack = rate_to_int[attack]
        defense = rate_to_int[defense]
    
    # so here we are creating two new columns and we initialize them in the apropriate manner ?
    p['attack_work_rate'] = attack
    p['defense_work_rate'] = defense
    
    return p

# you apply a method onto the players matrix
players = players.apply(extract_work_rate, axis=1)
# you can decide to remove some columns by specifying the corresponding correct name of the column
players = players.drop(columns=['Work Rate'])
```

In the code below you may choose to fill in the missing values of the players matrix as follows : `players.columns[players.isnull().any()].tolist()`. This must a returns a list of columns where we have a missing value. Then we can describe the data with : `players['SprintSpeed'].describe()`. We can fill in the missingdata with the mean : `players.fillna(players.mean(), inplace=True)`. But this is the mean of the whole matrix ? Consider the following code which can be used to split the data into training and testing data :

```
players_athletic = players[athletic_skills]
# you select the 2 column to the last column
X_athletic = players_athletic.values[:,1:] 
# you select the first column
y_athletic = players_athletic.values[:,0]

# in the below 1 is used as the random seed
X_train_athletic, X_test_athletic, y_train_athletic, y_test_athletic = train_test_split(X_athletic, y_athletic, test_size=0.3, random_state=1)
```

We can implement the ridge method as follows :

```
# these are empty arrays
alphas = [] 
scores = []
# and this is a method which takes in two parameters
def hyper_ridge(X, y):
    best_alpha = 0
    # this is a float which corresponds to the lowest possible negative number 
    highest_score = float('-inf')
    for alpha in np.linspace(0,2000,2001):
        # you calculate the ridge regression youing 2001 different alpha parameters
        ridge_reg = Ridge(alpha=alpha)
        # we append the given alpha to the array 
        alphas.append(alpha)
        # here we specify the type of algorithm we want to use, here this is the ridge regression and we use also the corresponding 
        # training data
        curr_score = cross_val_score(ridge_reg, X_train_athletic, y_train_athletic, cv=5, scoring='neg_mean_squared_error').mean()
        # you add the corresponding value to the array 
        scores.append(curr_score)
        # you update the data to use the correct alpha and the correct highest score
        if curr_score > highest_score:
            highest_score = curr_score
            best_alpha = alpha
    return best_alpha

# we then find the best alpha using the ridge method
alpha_athletic = hyper_ridge(X_train_athletic, y_train_athletic)
# then we use it in the ridge method
ridge_reg_athletic = Ridge(alpha=alpha_athletic)
ridge_reg_athletic.fit(X_train_athletic, y_train_athletic)
```

Suppose wewantto display the data in a dataframe :

```
# the second parameter are the columns in the dataframe starting from the second column
# in the below it's life weare turning the column vetor into a row vector, where you only choose the columns except for the first.
weights_athletic = pd.DataFrame(ridge_reg_athletic.coef_.reshape((1,ridge_reg_athletic.coef_.size)), columns=athletic_skills[1:])
weights_athletic = weights_athletic.sort_values(by=0, axis=1, ascending=False)
weights_athletic
```

You can concatenate the two columns where we find the minimum and the maximum of each column as follows :

```
pd.concat([players_athletic.min().rename('min'), players_athletic.max().rename('max')], axis=1)
```

Here what you are doing is essentially taking the minimum and maximum of each column and putting this data into a separate column. We apply min-max scaling. Not quite sure what the following does :

```
players_values = players.values
min_max_scaler = preprocessing.MinMaxScaler()
players_values = min_max_scaler.fit_transform(players_values)
players = pd.DataFrame(players_values, columns=players.columns)
```
You can also decide to find the relative correlation coefficients given a pair of categories. We have : `players_athletic.corr()`. Now we want to find the correlations between the overall category and the other variables as follows :

```
# here below you select the first row and all the columns starting from the second column
correlations = players_athletic.corr().iloc[0][1:]
# you can sort the valuesin an ascendings or a desending order
correlations = correlations.sort_values(ascending=False)
correlations = pd.DataFrame(correlations).T
correlations
```

Let's study bootstrap confidence intervals :

```
num_records = y.shape[0]
bootstrap_errors = []
bootstrap_errors_athletic = []
for i in range(1000):
    # you randomly choose an element from within the vector of num_records
    train_indices = np.random.choice(range(num_records), num_records, replace=True)
    # this must be a method that allows you to find the indices which are not in the set, hence the method is called the set 
    # difference in 1 d space
    test_indices = np.setdiff1d(range(num_records), train_indices)
    X_train_b, y_train_b = X[train_indices], y[train_indices]
    X_test_b, y_test_b = X[test_indices], y[test_indices]
    ridge_reg.fit(X_train_b, y_train_b)
    # when you have trainedthe model with the fit, then you calculate the mean squared error on the test vectors
    bootstrap_errors.append(mean_squared_error(y_test_b, ridge_reg.predict(X_test_b)))
    
    X_train_b, y_train_b = X_athletic[train_indices], y_athletic[train_indices]
    X_test_b, y_test_b = X_athletic[test_indices], y_athletic[test_indices]
    ridge_reg_athletic.fit(X_train_b, y_train_b)
    bootstrap_errors_athletic.append(mean_squared_error(y_test_b, ridge_reg_athletic.predict(X_test_b)))
    
bootstrap_errors_sorted = np.sort(bootstrap_errors)
bootstrap_errors_sorted_athletic = np.sort(bootstrap_errors_athletic)
# this you need to write in order to be able to output to the console
print('95% CIs')
# when you have variables you put them in curly brackets and to specify the type you write two dots followed by type
print('First model: [{:f}, {:f}]'.format(bootstrap_errors_sorted_athletic[25], bootstrap_errors_sorted_athletic[975]))
print('Second model: [{:f}, {:f}]'.format(bootstrap_errors_sorted[25], bootstrap_errors_sorted[975]))
```

We also have the following code : 

```
mean_error_athletic = bootstrap_errors_sorted_athletic.mean()
mean_error = bootstrap_errors_sorted.mean()
deviation_athletic = bootstrap_errors_sorted_athletic - mean_error_athletic
deviation = bootstrap_errors_sorted - mean_error
xpos = [0,1]
# plotting a bar graph, then you include the height of the error , for one bar you have an array of two values where the first is 
# likely the lower bound, the second is the upper bound
fig, ax = plt.bar(xpos, [mean_error_athletic, mean_error], yerr=[[deviation_athletic[25], deviation_athletic[975]], [deviation[25], deviation[975]]], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(xpos, ('First Model', 'Second Model'))
plt.ylabel('Error')
plt.title('Comparison of the Two Models')
```

We now will work with the PySpark library and attempt to scale up the capacity of the code that we are running. We have :

```
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import numpy as np 
import pandas as pd

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.executor.memory', '12g'),  # find
                                   ('spark.driver.memory','4g'), # your
                                   ('spark.driver.maxResultSize', '2G') # setup
                                  ])
                                  
# create the session, in the builder you need to have some configuration settings that will allow you to set up the session
# the session may already have been created in which case you just get it, you don't create it
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# create the context from the spark session
sc = spark.sparkContext

# FIX for Spark 2.x
locale = sc._jvm.java.util.Locale
locale.setDefault(locale.forLanguageTag("en-US"))
```

Then you would need to load the data in a spark datafrane as follows :

```
reddit = spark.read.json("messages.json.gz")
score = spark.read.json("score.json.gz")
```

Then we need to import some data and include some more code as follows :

```
%run hw2_utils.py
reddit.printSchema()
score.printSchema()
```

Now we want to display the following table of reddit data, as follows :

```
# in the below you group by the category, then you count all the number of elements in each group and store this data in the new 
# column which is called total_posts. Then you want to count the number of distinct authors in each group, and you give this a new 
# alias. Then you can find the average length of the post using the avg command.
# then you can place all this data in the cache by writing .cache()

subreddit_info = reddit.groupBy("subreddit")\
    .agg(count("*").alias("total_posts"), 
         countDistinct("author").alias("users_count"),
         avg(length("body")).alias("posts_length"),
         stddev(length("body")).alias("posts_length_stddev")
        ).cache()

subreddit_info
```
We display the largest subreddits in terms of total post count and the number of users :

```
# we select the two columns and we sort according to the descending order for the total_posts column. We convert the table to a 
# Pandas object
by_posts = subreddit_info.select("subreddit", "total_posts").sort(col("total_posts").desc()).toPandas()
by_users = subreddit_info.select("subreddit", "users_count").sort(col("users_count").desc()).toPandas()

display_side_by_side(by_posts, by_users)
```

Now we perform some selections using Spark :

```
import math

# We take the dataframe and we convert this to a Pandas object. We now sort the data in a descending order according to the post length
subreddits_by_pl = subreddit_info.toPandas().sort_values("posts_length", ascending=False).reset_index(drop=True)

# means you apply the anonymous function on each entry of the table
sqrt_N = subreddits_by_pl['total_posts'].apply(lambda r: math.sqrt(r))
# assigning the whole column to variable s 
s = subreddits_by_pl['posts_length_stddev']
# it would seem like here we are doing component wise operations
subreddits_by_pl['ci99'] = 2.576*(s / sqrt_N)

# horizontal bar chart and the first parameter is for the vertical axis, second for the horizontal axis
# the third parameter is the length of the error for each bar 
plt.barh(subreddits_by_pl.subreddit, subreddits_by_pl.posts_length, xerr=subreddits_by_pl.ci99)
plt.xlabel('Post length average (CI 99%)')
plt.ylabel('Subreddit')
plt.title('Average posts length')
plt.show()
```

Print the list of subreddits sorted by their average content scores :

```
# where we join the whole json file with this additionsal column
# where you group by one category, then you take the average of the scores, place into one new column called score
# then you need to sort all this data in a descending order according to the score column
# then you need to show this table
reddit.join(score, "id")\
    .groupBy("subreddit").agg(avg("score").alias("score"))\
    .sort(col("score").desc()).show()
```

Now we want to compute the most frequent words across the subreddits. We have :

```
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col

# you use the regex tokenizer to transform the table which we will call `reddit`
# must be that you split the words in that text by separating them against the white spaces 
regexTokenizer = RegexTokenizer(inputCol="body", outputCol="all_words", pattern="\\W")
reddit_with_words = regexTokenizer.transform(reddit)

# remove stop words
remover = StopWordsRemover(inputCol="all_words", outputCol="words")
# means that we remove the column all_words
reddit_with_tokens = remover.transform(reddit_with_words).drop("all_words")
reddit_with_tokens.show(5)

# get all words in a single dataframe, the way you do it is that you take the words column which contains arrays of the words present, 
# then you explod this, meaning that you create a separate column containing consecutively all the elements as separate entities.
all_words = reddit_with_tokens.select(explode("words").alias("word"))
# group by, sort and limit to 50k. What we do here is we take the table, we group by the word itself, and count how many times this 
# word appears in the column. Then we sort this column in descending order. 
top50k = all_words.groupBy("word").agg(count("*").alias("total")).sort(col("total").desc()).limit(50000)

top50k.show()
```
