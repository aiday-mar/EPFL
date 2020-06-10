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
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
ax.set_xlabel('Original')
ax.set_ylabel('Predicted')
plt.show()
mean_squared_error(y, predicted)
```
