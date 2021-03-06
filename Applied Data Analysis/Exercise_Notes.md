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

We get the subreddit representation as follows :

```
# Get all the words in a subreddit, and here you only select distinct words from this set and put into a column
tokens = reddit_with_tokens.select("subreddit", explode("words").alias("word")).distinct()
# Join with the whitelist of the top 50k, where you check where words are similar in the two columns, and you then select everything 
# in the tokens column
filtered_tokens = tokens.alias("t").join(top50k, tokens.word==top50k.word).select("t.*")

filtered_tokens
```
Now we need use the mapping and reducing method as follows :

```
# each row of the table r, we map to the property r.subreddit and the array formed with the r.word
subreddit_50k = filtered_tokens.rdd.map(lambda r: (r.subreddit, [r.word])).reduceByKey(lambda a,b: a+b).collect()

# for each element of the column
for sr in subreddit_50k:
    # we place {} in order to denote some variable goes in there
    print("Subreddit: {} - Words: {}".format(sr[0], len(sr[1])))
````

Let's compute the jaccard similarity:

```
# Note: similarity is computed 2 times! It can be optimized
similarity = []
for sr1 in subreddit_50k:
    for sr2 in subreddit_50k:
        # append the names, then have this followed by the jaccard similarity between the second compoenents
        similarity.append((sr1[0], sr2[0], jaccard_similarity(sr1[1], sr2[1])))

# make a dataframe from the list, index implies the row element, columns implies the column index, the corresponding value is the 
# third component
similarity_matrix_50k_words = pd.DataFrame(similarity).pivot(index=0, columns=1, values=2)
plot_heatmap(similarity_matrix_50k_words)
```

Alternatively we compute the 1000 most frequent words for each subreddit. 

```
# for each row r we  have the name of the category, one word present there and the one after
words_count_by_subreddit_rdd = reddit_with_tokens.rdd\
    .flatMap(lambda r: [((r.subreddit, w), 1) for w in r.words])\
    .reduceByKey(lambda a,b: a+b).cache()

# conversion in a dataframe, the dataframe is created after the reduction step above. For each r we create a row.
words_count_by_subreddit = spark.createDataFrame(
            # so the first [0] implies the first element and the second [0] implies the first element of the first element
            words_count_by_subreddit_rdd.map(lambda r: Row(subreddit=r[0][0], word=r[0][1], count=r[1]))
)

# Window on the words grouped by subreddit
# when you use the orderBy command, you need to specify the column specifically using the `col` keyword
window = Window.partitionBy(words_count_by_subreddit['subreddit']).orderBy(col('count').desc())

# Add position with rank() function (rowNumber is accepted, and it would be more correct)
# what does the rank method rank against ? 
top1000_rdd = words_count_by_subreddit.select('*', rank().over(window).alias('rank'))\
  .filter(col('rank') <= 1000).rdd.map(lambda r: (r.subreddit, [r.word])).reduceByKey(lambda a,b: a+b)
  # in the above we have the category followed by the list of words associated 

top1000 = top1000_rdd.collect()
```

We want to shoe the person who has the most answered or posted :

```
# we use the distinct keyword because it is sufficient the person posts once in the subreddit
# then to that specific person you assign one more 1. Then when you find the number of subreddits each author has posted in you
# apply the map function again where this time you create a new map but this time you need to create a row with an author and the 
# number of subreddits/communitiies he is part of 
user_in_communities_rdd = reddit.select("subreddit", "author").distinct().rdd.map(lambda r: (r.author, 1))\
    .reduceByKey(lambda a,b: a+b).map(lambda r: Row(author=r[0], communities=r[1]))

# Create a dataframe, this dataframe is created when we already mapped using the Row keyword
user_in_communities = spark.createDataFrame(user_in_communities_rdd).sort(col("communities").desc())

user_in_communities.show(1)
```

Then we draw a graph as follows :

```
users = reddit.select("subreddit", "author").distinct()\
    .rdd.map(lambda r: (r.subreddit, [r.author]))\
    .reduceByKey(lambda a,b: a+b).collect()

similarity_users = []
for sr1 in users:
    for sr2 in users:
        similarity_users.append((sr1[0], sr2[0], jaccard_similarity(sr1[1], sr2[1])))
        
similarity_matrix_users = pd.DataFrame(similarity_users).pivot(index=0, columns=1, values=2)

# we use nba subreddit
sr = "nba"

fig, ax = plt.subplots(figsize=(10,8))

# means we take the row for the nba subreddit, then when we have the row, we drop the column/category sr and take the zeroth element 
# which is now the jaccard value for the users
u = similarity_matrix_users[similarity_matrix_users.index == sr].drop(sr, axis=1).to_numpy()[0]
w = similarity_matrix_words[similarity_matrix_words.index == sr].drop(sr, axis=1).to_numpy()[0]
ax = plt.scatter(w, u)

# now we have the row with all the jaccard values, and you want to take the names of the words which are given by the column names
c=similarity_matrix_users[similarity_matrix_users.index == sr].drop(sr, axis=1).columns.tolist()

for i in range(0, len(u)):
    # you associated the ith element of the vector c with the point xy as seen below. You associated a rotation, so the text is
    # rotated
    plt.annotate(c[i], xy=(w[i], u[i]), rotation=20)
    
plt.xlabel("Words Jaccard")
plt.ylabel("Users Jaccard")
plt.title("NBA similarity")
```

**Tutorial 7**

To run the notebook for the tutorial we will need to install the following : 

```
conda install nltk gensim spacy
pip install pyLDAvis
pip install vaderSentiment
pip install empath
python -m spacy download en
python -m nltk.downloader punkt
python -m nltk.downloader all-corpora 
```
Then we need to need to import some libraries that will we be used througout the code : 

```
%load_ext autoreload
# Reload all modules (except those excluded by %aimport) every time before executing the Python code typed.
%autoreload 2

import warnings; warnings.simplefilter('ignore')
import os, codecs, string, random
import numpy as np
from numpy.random import seed as random_seed
from numpy.random import shuffle as random_shuffle
import matplotlib.pyplot as plt
%matplotlib inline  

seed = 42
random.seed(seed)
np.random.seed(seed)

#NLP libraries
import spacy, nltk, gensim, sklearn
import pyLDAvis.gensim

#Vader
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Scikit imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

#The data
corpus_root = 'books/'
```

we intialize the natural language processing helper using the spacy library which will analyze English text as follows : 

```
nlp = spacy.load('en')
```

Then after this we need to load the books as follows :

```
books = list()

for book_file in os.listdir(corpus_root):
    if ".txt" in book_file:
        print(book_file)
        with codecs.open(os.path.join(corpus_root,book_file),encoding="utf8") as f:
            books.append(f.read())
```

Here we are creating first a list called books. Then what we do is access the folder called books, then we make this folder into a list directory, the command is in the os library. Then we iterate over the files in this directory. Then if this file is a text file, then we print the file. Then we get the full name of the path of the book by joining the path of the root and the name of the book. For this we use a specific encoding. Then given this filename path we can actually open the file and give it the name f. Then to this list of books we actually add the file f which in this case we would have read. Then we can print the begining of the third file, where we access the third file with [3] then we can access the lines with the second pair of brackets. Hence :

```
print(books[3][0:600])
```

Now suppose that we would want to remove the extra blank lines used to create new paragpraphs then you can write : 

```
books = [" ".join(b.split()) for b in books]
```

Here you bsically say that the book's words are split and then we join then with just one little space. Then you can choose to perform the language processing on one of the books as follows : 

```
#index all books
book = books[1]

#put in raw text, get a Spacy object
doc = nlp(book)
```

You input the book into the spacy library. You may then want to split the text against its lines as follows : 

```
# firs we create a list of sentences, where we can access the sentences with the command sent associated to the nlp model
sentences = [sent for sent in doc.sents]
print('Sentence 1:',sentences[0],'\n')
print('Sentence 2:',sentences[1],'\n')
print('Sentence 3:',sentences[2],'\n')
```
Then instead you may want to split the example into tokens. We have :

```
example = 'I am already far north of London, and as I walk in the streets of Petersburgh, I feel a cold northern breeze play upon my cheeks, which braces my nerves and fills me with delight.'

doc = nlp(example)

#strings are encoded to hashes
tokens = [token.text for token in doc]

print(example,'\n')
print(tokens)
```

First there is a string. Then we apply the model on the string which creates a document. Then you can access individually the tokens in the document, and to individually get the text we can write : `token.text`. Then when you want to print the array of tokens you can write : `print(tokens)`. You can also decide to tag the words in order to get the word itself followed by the type of the word : 

```
# Here we have the text of the token followed by the type, which is either a verb, a pronoun, an adverb etc
pos_tagged = [(token.text, token.pos_) for token in doc]

print(example,'\n')
print(pos_tagged)
```
In a similar way we have `token.label_` designates that we are dealing with a category of objects. Now we can select the first ten elements of the list as follows : `list(spacy_stopwords)[:10]`. Here we are first selecting the first 10 elements, and then we convert this to a list. You can detect the stop words in a sentence as follows : 

```
print(example,'\n')
stop_words = [token.text for token in doc if token.is_stop]
print(stop_words)
```

Noun chunks are "base noun phrases" – flat phrases that have a noun as their head -- a noun plus the words describing the noun – for example, "the lavish green grass" or "the world’s largest tech fund". We can print them as :

```
print(example,'\n')

for chunk in doc.noun_chunks:
    print(chunk.text)
```

You can count word occurences as follows :

```
from collections import Counter

print(example,'\n')
words = [token.text for token in doc]

# where here we apply the counter to the list of tokens in the document
word_freq = Counter(words)
# then we try to find the most common words
common_words = word_freq.most_common()

print(common_words)
```

Take up the example above and now suppose we want to remove the stop word as well as the punctuation and print out the same type of list.

```
# meaning we select the token texts in the given document when the token is not a stop word nor a form of punctuation
words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]

# five most common tokens
word_freq = Counter(words)
common_words = word_freq.most_common()

print(common_words)
```

You may at some point decide to not load a certain component of the pipeline by disabling the appropriate components of the pipeline : 

```
nlp.remove_pipe('parser')
nlp.remove_pipe('tagger')
```

To deal with sentiment analysis you can first initialize the analyzer as follows : 

```
analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores(example)
```

You can use this analyzer as follows :

```
print(example, '\n')
print('Negative sentiment:',vs['neg'])
print('Neutral sentiment:',vs['neu'])
print('Positive sentiment:',vs['pos'])
print('Compound sentiment:',vs['compound'])
```

You can decide to see what the positivity level of the book pride and prejudice is when you analyze the story sentence by sentence as follows :

```
# where here we decide to use the spacy library 
nlp = spacy.load('en')
# and then we input there the 3rd book into the model
doc = nlp(books[3])
# initialized to be empty 
positive_sent = []
#iterate through the sentences, get polarity scores
# here we add into the list for all the sentences in the document, the polarity score given the text of the sentence.
# where here you evaluate the level of positivity. 
[positive_sent.append(analyzer.polarity_scores(sent.text)['pos']) for sent in doc.sents]
plt.hist(positive_sent,bins=15)
plt.xlim([0,1])
plt.ylim([0,8000])
plt.xlabel('Positive sentiment')
plt.ylabel('Number of sentences')
```
Here we do the same by the compound category, meaning that we evaluate the feelings overall negative and positive from the sentences. To each sentence we actually associate a score between -1 and 1. In this way we can find the number of positive and negative sentences as follows :

```
# where here we have an analyzer, and then we can analyse the polarity scores in a way using the text of the sentence
sents = [analyzer.polarity_scores(sent.text)['compound'] for sent in doc.sents]
# where here we select the entries in the sents where the value is bigger than 0.05. Then you sum the number of such entries. 
print('Number of positive sentences:',sum(np.array(sents)>=0.05))
print('Number of negative sentences:',sum(np.array(sents)<=-0.05))
print('Number of neutral sentences:',sum(np.abs(np.array(sents))<0.05))
```

But I thought that the above sum would actually sum the values in the corresponding elements not sum the number of times the condition is verified. Now we will perform document classification as follows :

```
# Let's load our corpus via NLTK this time, we have this method as specified below 
from nltk.corpus import PlaintextCorpusReader
?PlaintextCorpusReader
# here we are taking the name of the folder and selecting all the files having a txt extension 
our_books = PlaintextCorpusReader(corpus_root, '.*.txt')
# here you can show the names of the files in the given list by printing it 
print(our_books.fileids())
```

Here we segment the books into equally long chunks.

```
def get_chunks(l, n):
    """Yield successive n-sized chunks from l where l is a list"""
    # here below we find the range of indices from 0 to the length of the list l and splint into chunks of size n
    for i in range(0, len(l), n):
        # then each time this returns us the sublist of the list l which starts at index i and where we go up to i+n
        # however what if the length of the list is not divisble by n ? 
        yield l[i:i + n]


# we select the files f and n the name of the book, by using the enumerate function on the list of the names of the books.
# so in this list it would seem like we have the name of the book followed by the actual file of the book 
book_id = {f:n for n,f in enumerate(our_books.fileids())} # dictionary of books

chunks = list()
chunk_class = list() # this list contains the original book of the chunk, for evaluation

limit = 500 # how many chunks total
size = 50 # how many sentences per chunk/page

for f in our_books.fileids():
    # this sentence below returns all the sentences associated to the file f 
    sentences = our_books.sents(f)
    # then you print the name of the file followed by two dots
    print(f,":")
    print('Number of sentences:',len(sentences))
    
    # create chunks on the list of sentences, where the chunks are of size size 
    chunks_of_sents = [x for x in get_chunks(sentences,size)] 
    # this is a list of lists of sentences, which are a list of tokens
    chs = list()
    
    # regroup so to have a list of chunks which are strings
    for c in chunks_of_sents:
        grouped_chunk = list()
        # for each chunk in the list of chunks of sentences. 
        for s in c:
            grouped_chunk.extend(s)
        # where here we are appending each group/chunk separately into this list 
        chs.append(" ".join(grouped_chunk))
    # the below code outputs the number of chunks and then we add a new line 
    print("Number of chunks:",len(chs),'\n')
    
    # filter to the limit, to have the same number of chunks per book, meaning we only choose the first limit chunks of the book
    chunks.extend(chs[:limit])
    # where we have the length of the vector of chs[:limit] and then we create a vector from 1 to this length
    # what is this underscore ? and why are we using the book id of the file to iterate ? 
    chunk_class.extend([book_id[f] for _ in range(len(chs[:limit]))])
```

Then we represent the chunks with bags of words, as follows :

```
vectorizer = CountVectorizer()

# initialize and specify minumum number of occurences to avoid untractable number of features
# vectorizer = CountVectorizer(min_df = 2) if we want high frequency

# create bag of words features
X = vectorizer.fit_transform(chunks)

# meaning first we find the number of rows and then we find the number of columns
print('Number of samples:',X.toarray().shape[0])
print('Number of features:',X.toarray().shape[1])

# mask and convert to int Frankenstein
# what does the code below do ? 
Y = np.array(chunk_class) == 1
Y = Y.astype(int)  

#shuffle the data
# seems like you start from a random state to initialize the randomization
X, Y = shuffle(X, Y, random_state=0)

#split into training and test set
# the rest of the 0.8 fractional part is the training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

Let's fit the regularized logistic regression. Here we crossvalidate the regularization parameter on the training set. 

```
accs = []

#the grid of regularization parameter 
grid = [0.01,0.1,1,10,100,1000,10000]

for c in grid:
    
    #initialize the classifier, what does the solver value stand for ? 
    clf = LogisticRegression(random_state=0, solver='lbfgs',C = c)
    
    #crossvalidate, where the first parameter clf designates the type of regrssion we are using 
    scores = cross_val_score(clf, X_train,Y_train, cv=10)
    accs.append(np.mean(scores))

plt.plot(accs)
plt.xticks(range(len(grid)), grid)
plt.xlabel('Regularization parameter \n (Low - strong regularization, High - weak regularization)')
plt.ylabel('Crossvalidation accuracy')
plt.ylim([0.986,1])
```

Then we train on the training set and test on the testing set. 

```
# here we fit a curve where we use the X_train and the Y_train sets of data
clf = LogisticRegression(random_state=0, solver='lbfgs',C = 10).fit(X_train,Y_train)

#predict on the test set, using the score method and using the method of the logistic regression. 
print('Accuracy:',clf.score(X_test,Y_test))
```

Don't quite understand the code below :

```
# here we take the first coefficient of the logistic regression
coefs=clf.coef_[0]

# numpy.argpartition() function is used to create a indirect partitioned copy of input array with its elements rearranged in 
# such a way that the value of the element in k-th position is in the position it would be in a sorted array. All elements 
# smaller than the k-th element are moved before this element and all equal or greater are moved behind it. The ordering of the 
# elements in the two partitions is undefined.It returns an array of indices of the same shape as arr, i.e arr[index_array] 
# yields a partition of arr.
top_three = np.argpartition(coefs, -20)[-20:]

# why does it say top three but it prints out more than three
print(np.array(vectorizer.get_feature_names())[top_three])

print(example,'\n')
# then you take the model, you evaluate it in the vector, you change this result to the vector and then you select the first 11
# elements, this is transformed into a list 
print('Embedding representation:',list((nlp(example).vector)[0:10]),'...')
```

Now suppose that we want to do some topic detection, then we have :

```
# Get the chunks again (into smaller chunks), where once again we must have the name of the book followed by the actual file 
book_id = {f:n for n,f in enumerate(our_books.fileids())} # dictionary of books
chunks = list()
chunk_class = list() # this list contains the original book of the chunk, for evaluation

limit = 60 # how many chunks total
size = 50 # how many sentences per chunk/page

for f in our_books.fileids():
    # where we have the sentences are the ones associated with the different files in the folder 
    sentences = our_books.sents(f)
    # then we print the file f
    print(f)
    print('Number of sentences:',len(sentences))
    
    # create chunks of the list of sentences with the given size, then we iterate over all chunks of this, and made into list
    chunks_of_sents = [x for x in get_chunks(sentences,size)] 
    # this is a list of lists of sentences, which are a list of tokens
    chs = list()
    
    # regroup so to have a list of chunks which are strings
    for c in chunks_of_sents:
        grouped_chunk = list()
        for s in c:
            # probably means we are adding this chunk to the list of grouped chunks
            grouped_chunk.extend(s)
        chs.append(" ".join(grouped_chunk))
        # you append the chunks and remove the spaces in between
    print("Number of chunks:",len(chs),'\n')
    
    # filter to the limit, to have the same number of chunks per book
    # maybe extend is like append but for multiple elements
    chunks.extend(chs[:limit])
    # what is the below 
    chunk_class.extend([book_id[f] for _ in range(len(chs[:limit]))])

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

processed_docs = list()
for doc in nlp.pipe(chunks, n_threads=5, batch_size=10):

    # Process document using Spacy NLP pipeline.
    ents = doc.ents  # Named entities

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Remove common words from a stopword list and keep only words of length 3 or more.
    doc = [token for token in doc if token not in STOPWORDS and len(token) > 2]

    # Add named entities, but only if they are a compound of more than word. Converting here the entity into a string
    # and making sure the length of the entity is bigger than one 
    doc.extend([str(entity) for entity in ents if len(entity) > 1])

    processed_docs.append(doc)
docs = processed_docs
# once we copied the contents of the processed_docs into the simple doc we delete the initial variable 
del processed_docs

# Add bigrams too
from gensim.models.phrases import Phrases

# does this mean phrases that appear at least 15 times ? 
# Add bigrams to docs (only ones that appear 15 times or more).
bigram = Phrases(docs, min_count=15)

for idx in range(len(docs)):
    # we get the idx element of the docs and then pass it to the bigram ? 
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# Create a dictionary representation of the documents, and filter out frequent and rare words.
from gensim.corpora import Dictionary
dictionary = Dictionary(docs)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
# can't appear more frequently than this 
max_freq = 0.5
# the number of times the word should appear
min_wordcount = 5
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

# Bag-of-words representation of the documents, where we use the abbreviation Bag of words BOw.
corpus = [dictionary.doc2bow(doc) for doc in docs]
#MmCorpus.serialize("models/corpus.mm", corpus)

print('Number of unique tokens: %d' % len(dictionary))
print('Number of chunks: %d' % len(corpus))

# models
from gensim.models import LdaMulticore
# below we have a word associated to a corresponding value 
params = {'passes': 10, 'random_state': seed}
base_models = dict()

model = LdaMulticore(corpus=corpus, num_topics=4, id2word=dictionary, workers=6,
                passes=params['passes'], random_state=params['random_state'])

# likely means that for each topic, you display 5 associated words, where as above you have 4 topics 
model.show_topics(num_words=5)
# here you show 20 potential topics
model.show_topic(1,20)

# shown in reverse order, and then the key is given by the first sublist of x ? why do we use a lambda function for the key to sort against
# here we are sorting the output when corpus[0] is put into the model 

# Key is a function that will be called to transform the collection's items before they are compared. 
# The parameter passed to key must be something that is callable.
# The use of lambda creates an anonymous function (which is callable). In the case of sorted the callable only takes one 
# parameters. Python's lambda is pretty simple. It can only do and return one thing really.
# The syntax of lambda is the word lambda followed by the list of parameter names then a single block of code. 
# The parameter list and code block are delineated by colon. This is similar to other constructs in python as well such as 
# while, for, if and so on. They are all statements that typically have a code block. Lambda is just another instance of a 
# statement with a code block.
sorted(model[corpus[0]],key=lambda x:x[1],reverse=True)

# plot topics using the given model, the right body and dictionary
data =  pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.display(data)

# assignment
# we initially write an empty list
sent_to_cluster = list()
# inside this body we have two columns, we are enumerating over
for n,doc in enumerate(corpus):
    # likely means that if the given document exists, then we run the following commands that follow 
    if doc:
        # are we taking the maximum when we evaluate the document with the model and when we are using that specific key
        # essentially taking the first column/row? 
        cluster = max(model[doc],key=lambda x:x[1])
        sent_to_cluster.append(cluster[0])

# accuracy
from collections import Counter
# because we have that in the list of items we have a book which is associated to a cluster
for book, cluster in book_id.items():
    assignments = list()
    # The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed iterator 
    # is paired together, and then the second item in each passed iterator are paired together etc.
    for real,given in zip(chunk_class,sent_to_cluster):
        if real == cluster:
            assignments.append(given)
    # the counter class allows you to find the most common element in a list for example, where the 1 indicates the top most
    # common element, the [0] indicates you are considering only the first row of the Counter ? 
    most_common,num_most_common = Counter(assignments).most_common(1)[0] # 4, 6 times
    print(book,":",most_common,"-",num_most_common)
    print("Accuracy:",num_most_common/limit)
    print("------")
```

Now we hve the next task which is the semantic analysis based on lexical categories. Consider the following code : 

```
from empath import Empath
lexicon = Empath()

# here we take the Empath instance, take the categories and find the different keys, you need to iterate over the keys, not
# the column name which is presumably cats in this case
for cat in lexicon.cats.keys():
    print(cat)

# Next we can display all the elements that are in the in health category of the lexicon
lexicon.cats["health"]
```

Now let us use this Empath library in order to study pride and prejudice. First we use the spacy library as follows : 

```
nlp = spacy.load('en')
doc = nlp(books[3])
# you analyze the text of that specific document where you use the apropriate categories too and you also need to 
# normalize the text
empath_features = lexicon.analyze(doc.text,categories = ["disappointment", "pain", "joy", "beauty", "affection"], normalize = True)
bins = range(0,len(doc.text),150000)

# now we want to showcase the evolution of topics where we have chosen 4 random topics
love = []
pain = []
beauty = []
affection = []

# meaning we are going to enumerate over the whole vector bins all the way to the last
for cnt,i in enumerate(bins[:-1]):
    # then we analyze the text from the given range, and we input the categories that interest us 
    empath_features = lexicon.analyze(doc.text[bins[cnt]:bins[cnt+1]],
                                      categories = ["love", "pain", "joy", "beauty", "affection"], normalize = True)
    love.append(empath_features["love"])
    pain.append(empath_features["pain"])
    beauty.append(empath_features["beauty"])
    affection.append(empath_features["affection"])

# you can decide to plot the categories on a graph in order to be able to see how the topics develop throughout
# the story
plt.plot(love,label = "love")
plt.plot(beauty, label = "beauty")
plt.plot(affection, label = "affection")
plt.plot(pain,label = "pain")

plt.xlabel("progression in the book")
plt.ylabel("frequency of a category")
plt.legend()
```

**Tutorial 08**

In this tutorial we are going to study how to apply python to networks and the use in data science. First we need to import the apropriate libraries as follows :

```
import networkx as nx

import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import collections
from community import community_louvain
from networkx.algorithms.community.centrality import girvan_newman
import itertools
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```

You can display the version of the library you are using by writing : `nx.__version__`. Then you want to install the apropriate libraries as follows :

```
conda install -c anaconda pandas
conda install -c conda-forge matplotlib 
pip install networkx==2.3.0 python-louvain
```

First we construct a simple undirected graph :

```
G = nx.Graph() # for a directed graph use nx.DiGraph()
G.add_node(1)
# because you can add nodes from a vector of numbers 
G.add_nodes_from(range(2,9))  # add multiple nodes at once

# add edges 
G.add_edge(1,2)
# the edges are esentially pairs of nodes 
edges = [(2,3), (1,3), (4,1), (4,5), (5,6), (5,7), (6,7), (7,8), (6,8)]
G.add_edges_from(edges)
G.nodes()

# allows in essence to describe the graph
print(nx.info(G))
```

There is also a plotting engine as follows : `nx.draw_spring(G, with_labels=True,  alpha = 0.8)`, the parameters present are the following, the graph G, the boolean with_labels set to true and the alpha value. Below we introduce some helper functions which we will be using : 

```
# Helper function for plotting the degree distribution of a Graph
def plot_degree_distribution(G):
    degrees = {}
    for node in G.nodes():
        # create a temporary variable containing the degree of the given node of the graph. 
        degree = G.degree(node)
        if degree not in degrees:
            degrees[degree] = 0
        degrees[degree] += 1
        # means that this vector displays how many nodes of degree d there are at node which has name index i
    # here we take the items of the list and we need to sort this in ascending or descending order 
    sorted_degree = sorted(degrees.items())
    # in this list os sorted degrees you display the pairs of keys and values 
    deg = [k for (k,v) in sorted_degree]
    cnt = [v for (k,v) in sorted_degree]
    # in this subplot you have a, or multiple figures, and you have an axis corresponding to it
    fig, ax = plt.subplots()
    # it will be a bar graph where you plot the degree against the frequency
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title("Degree Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Degree")
    ax.set_xticks([d+0.05 for d in deg])
    ax.set_xticklabels(deg)
```

Then you have the following code which is used to describe the graph :

```
# Helper function for printing various graph properties
def describe_graph(G):
    print(nx.info(G))
    # here we are checking whether the graph is connected or not 
    if nx.is_connected(G):
        # then you print after this line the actual average length path of the graph G
        print("Avg. Shortest Path Length: %.4f" %nx.average_shortest_path_length(G))
        print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
    else:
        print("Graph is not connected")
        print("Diameter and Avg shortest path length are not defined!")
    print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph
    # #closed-triplets(3*#triangles)/#all-triplets
    print("Global clustering coefficient aka Transitivity: %.4f" %nx.transitivity(G))
```

Finally we want to visualize the graph as follows : 

```
# Helper function for visualizing the graph, the node shape in this case is circular 
def visualize_graph(G, with_labels=True, k=None, alpha=1.0, node_shape='o'):
    # nx.draw_spring(G, with_labels=with_labels, alpha = alpha), must be a specific layout this layout 
    pos = nx.spring_layout(G, k=k)
    if with_labels:
        # meaning that the labels is a dictionary mapping node n to label n, where we iteate over the nodes n of the graph.
        lab = nx.draw_networkx_labels(G, pos, labels=dict([(n, n) for n in G.nodes()]))
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='g', node_shape=node_shape)
    plt.axis('off')
```

Suppose we want to create and visualize an Erdo-Renyi graph. This can be coded as :

```
n = 10  # 10 nodes
m = 20  # 20 edges

# where here you specified beforehand the number of nodes and the number of edges and then you create a random graph based on that
erG = nx.gnm_random_graph(n, m)

describe_graph(erG)
visualize_graph(erG, k=0.05, alpha=0.8)
plot_degree_distribution(erG)
```

Another example of working with a network can be found below. Here we have the following code :

```
data_folder = './data/quakers/'
nodes = pd.read_csv(data_folder + 'quakers_nodelist.csv')
nodes.head(10)

# where Gender is the name of a category, and we need to change the type of the category with the code below 
nodes.Gender = nodes.Gender.astype('category')
# First we write the initial name of the column : final name of the column
nodes = nodes.rename({'Historical Significance': 'Role'}, axis = 1)
print('There are ', len(nodes), 'quakers')

# check if the names are unique
len(nodes['Name'].unique()) == len(nodes)

# since the names are unique, index based on names
nodes.set_index('Name', inplace=True)

# let's also add a new attribute based on the role column. Is (s)he a directly involved Quaker or not? 
# meaning here we look at the string identifying the role of the person and we verify whether the string 'quaker' is in the role or not
nodes['Quaker'] = ['quaker' in role.lower() for role in nodes.Role]
nodes.head()
```
You can load the edge list from the pandas data frame as follows :

```
# the first parameter is the list of edges, we have the names of the columns that interest us
# there are no edge attributes and for this we will be using the nx library
quakerG =nx.from_pandas_edgelist(edges, 'Source', 'Target', edge_attr=None, create_using= nx.Graph())
describe_graph(quakerG)

# add node attributes by passing dictionary of type name -> attribute
nx.set_node_attributes(quakerG, nodes['Role'].to_dict(), 'Role' )
nx.set_node_attributes(quakerG, nodes['Gender'].to_dict(), 'Gender' )
nx.set_node_attributes(quakerG, nodes['Birthdate'].to_dict(), 'Birthdate' )
nx.set_node_attributes(quakerG, nodes['Deathdate'].to_dict(), 'Deathdate' )
nx.set_node_attributes(quakerG, nodes['Quaker'].to_dict(), 'Quaker' )

# You can easily get the attributes of a node
quakerG.node['William Penn']

# we can start by visually inspecting the graph
visualize_graph(quakerG, False, k=0.05, alpha=0.4)

# the code below allows you to find the number of connected components :
print(nx.is_connected(quakerG))
comp = list(nx.connected_components(quakerG))
print('The graph contains', len(comp), 'connected components')

#now we basically select the largest component, what is this key though ?
largest_comp = max(comp, key=len)
percentage_lcc = len(largest_comp)/quakerG.number_of_nodes() * 100
print('The largest component has', len(largest_comp), 'nodes', 'accounting for %.2f'% percentage_lcc, '% of the nodes')

# now suppose that you want to find the shortest path between two quackers, given these are in the same component
fell_whitehead_path = nx.shortest_path(quakerG, source="Margaret Fell", target="George Whitehead")
print("Shortest path between Fell and Whitehead:", fell_whitehead_path)

# take the largest component and analyse its diameter = longest shortest-path
# we take the subgraph which is given by the largest componentn
lcc_quakerG = quakerG.subgraph(largest_comp)
print("The diameter of the largest connected component is", nx.diameter(lcc_quakerG))
print("The avg shortest path length of the largest connected component is", nx.average_shortest_path_length(lcc_quakerG))
```

The global measure called transitivity (aka global clustering coefficient), is the ratio of all existing triangles (closed triples) over all possible triangles (open and closed triplets): `print('%.4f' %nx.transitivity(quakerG))`.

Employ a local measure called clustering coefficient, which quantifies for a node how close its neighbours are to being a clique (complete graph). Measured as the ratio of, the number of edges to the number of all possible edges, among the neighbors of a node. Here we have : `print(nx.clustering(quakerG, ['Alexander Parker', 'John Crook']))`.

```
# Lets check by looking at the subgraphs induced by Alex and John
subgraph_Alex = quakerG.subgraph(['Alexander Parker']+list(quakerG.neighbors('Alexander Parker')))
subgraph_John = quakerG.subgraph(['John Crook']+list(quakerG.neighbors('John Crook')))
nx.draw_spring(subgraph_Alex, with_labels=True)
nx.draw_circular(subgraph_John, with_labels=True)
```

To find the most important node you can check which one has the highest degree :

```
# the degrees for the nodes of the graph
degrees = dict(quakerG.degree(quakerG.nodes()))
# now you sort the list of all the degrees 
sorted_degree = sorted(degrees.items(), key=itemgetter(1), reverse=True)

# And the top 5 most popular quakers are.. 
for quaker, degree in sorted_degree[:5]:
    print(quaker, 'who is', quakerG.node[quaker]['Role'], 'knows', degree, 'people')

# meaning you choose the second element in the sorted set of degrees
degree_seq = [d[1] for d in sorted_degree]
degreeCount = collections.Counter(degree_seq)
# you create a dataframe rom the counter 
degreeCount = pd.DataFrame.from_dict(degreeCount, orient='index').reset_index()
fig = plt.figure()
ax = plt.gca()
# you plot two different columns from the table against each other 
ax.plot(degreeCount['index'], degreeCount[0], 'o', c='blue', markersize= 4)
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.title('Degree distribution for the Quaker network')

# As a bar plot
plot_degree_distribution(quakerG)
```
Suppose that we want to implement code to display the katz centrality : 

```
# here you make a dictionary between the nodes of the graph and their respective degrees 
degrees = dict(quakerG.degree(quakerG.nodes()))

# you assign the katz centrality respectively to each node 
katz = nx.katz_centrality(quakerG)
nx.set_node_attributes(quakerG, katz, 'katz')
sorted_katz = sorted(katz.items(), key=itemgetter(1), reverse=True)

# And the top 5 most popular quakers are.. 
for quaker, katzc in sorted_katz[:5]:
    print(quaker, 'who is', quakerG.node[quaker]['Role'], 'has katz-centrality: %.3f' %katzc)
```
In a similar manner you can measure the betweeness centrality :
```
# Compute betweenness centrality
betweenness = nx.betweenness_centrality(quakerG)
# Assign the computed centrality values as a node-attribute in your network
nx.set_node_attributes(quakerG, betweenness, 'betweenness')
sorted_betweenness = sorted(betweenness.items(), key=itemgetter(1), reverse=True)

for quaker, bw in sorted_betweenness[:5]:
    print(quaker, 'who is', quakerG.node[quaker]['Role'], 'has betweeness: %.3f' %bw)
```

Now we plot this betweeness centrality :

```
# similar pattern
list_nodes =list(quakerG.nodes())
list_nodes.reverse()   # for showing the nodes with high betweeness centrality 
# we have chosen a specific layout for the graph 
pos = nx.spring_layout(quakerG)
ec = nx.draw_networkx_edges(quakerG, pos, alpha=0.1)
# first yu access the node then the value of the betweeness for the different nodes. 
nc = nx.draw_networkx_nodes(quakerG, pos, nodelist=list_nodes, node_color=[quakerG.nodes[n]["betweenness"] for n in list_nodes], with_labels=False, alpha=0.8, node_shape = '.')

# why do we use here the colorbar method ? 
plt.colorbar(nc)
plt.axis('off')
plt.show()
```

Girvan Newman idea : Edges possessing high betweeness centrality separate communities. Let's apply this on our toy sample graph (G) to get a better understanding of the idea. The algorithm starts with the entire graph and then it iteratively removes the edge with the highest betweeness.

```
comp = girvan_newman(G)
it = 0
# here we slice the comp into 4 parts then we print the right data
for communities in itertools.islice(comp, 4):
    it +=1
    print('Iteration', it)
    print(tuple(sorted(c) for c in communities))
visualize_graph(G,alpha=0.7)
```

The louvain method : It proceeds the other way around. Initially every node is considered as a community. The communities are traversed, and for each community it is tested whether by joining it to a neighboring community, we can obtain a better clustering.

```
partition = community_louvain.best_partition(quakerG)
# add it as an attribute to the nodes
for n in quakerG.nodes:
    quakerG.nodes[n]["louvain"] = partition[n]

# plot it out
pos = nx.spring_layout(quakerG,k=0.2)
ec = nx.draw_networkx_edges(quakerG, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(quakerG, pos, nodelist=quakerG.nodes(), node_color=[quakerG.nodes[n]["louvain"] for n in quakerG.nodes], with_labels=False, node_size=100, cmap=plt.cm.jet)
plt.axis('off')
plt.show()

# which is the name of the one of the partitions
cluster_James = partition['James Nayler']
# Take all the nodes that belong to James' cluster
members_c = [q for q in quakerG.nodes if partition[q] == cluster_James]
# get info about these quakers
for quaker in members_c:
    print(quaker, 'who is', quakerG.node[quaker]['Role'], 'and died in ',quakerG.node[quaker]['Deathdate'])

# this is another of the names of the parts of the partition
cluster_Lydia = partition['Lydia Lancaster']
# Take all the nodes that belong to Lydia's cluster
members_c = [q for q in quakerG.nodes if partition[q] == cluster_Lydia]
# get info about these quakers
for quaker in members_c:
    print(quaker, 'who is', quakerG.node[quaker]['Role'], 'and died in ',quakerG.node[quaker]['Deathdate'])

nx.attribute_assortativity_coefficient(quakerG, 'Gender')
```
