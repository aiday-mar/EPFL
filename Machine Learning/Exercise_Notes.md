# Exercise Notes

**Problem set 10**

We fill in the notebook function split_data to split the dataset into training and testing data. We need to import the right libraries that we will use to analyse the data.

```
%matplotlib inline
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
```

Then we need to use the helpers library in order to import the appropriate CSV file, the content of which we will be using hereby in our data analysis.

```
from helpers import load_data, preprocess_data
path_dataset = "movielens100k.csv"
ratings = load_data(path_dataset)
```

Now we use th plots library and import the plot_raw_data function. We have :

```
from plots import plot_raw_data

num_items_per_user, num_users_per_item = plot_raw_data(ratings)

print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))
```

Now we have the ratings are plotted and the resulting data is put into two column vectors which are num_items_per_user and num_users_per_item. Next when we have a variable we can put these curly brackets instead to replace the variable {}. Then to place the data we use the .format() function. 
