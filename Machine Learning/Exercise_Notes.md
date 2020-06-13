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


The actual splitting into training and testing code is as follows :

```
def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):

# set seed
np.random.seed(988)

# select user and item based on the condition.
# We have this where method part of the numpy library and we have the vector bigger than the min_num_ratings.
# The first column has index 0, [0]
valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
# So we have the rating matrix where we choose the elements where we have chosen the valid_terms and the valid_users on the horizontal axis
valid_ratings = ratings[valid_items, :][: , valid_users]  

# where the valid_ratings element is a matrix and has two dimensions
num_rows, num_cols = valid_ratings.shape
# in the scipy sparse library we use the lil_matrix method.
train = sp.lil_matrix((num_rows, num_cols))
test = sp.lil_matrix((num_rows, num_cols))

print("the shape of original ratings. (# of row, # of col): {}".format(
ratings.shape))
print("the shape of valid ratings. (# of row, # of col): {}".format(
(num_rows, num_cols)))

# so in the valid_ratings you choose also the non zero elements, and hence you find the indices of the items and users where there are # non zero elements
nz_items, nz_users = valid_ratings.nonzero()

# split the data
for user in set(nz_users):
        # for each valid user in nz_users, we check over the elements in the corresponding row and find where it is non_zero, and 
        # therefore find the indices of the rows which have non zero element        
        row, col = valid_ratings[:, user].nonzero()
        # we select as the size the integer corresponding to the fractional part of the row where the fraction is specifically p_test
        selects = np.random.choice(row, size=int(len(row) * p_test))
        # this notation means that we take the sets of row indices and remove the set of selects indices leaving us with the residual 
        # indices. 
        residual = list(set(row) - set(selects))

        # Here you define the train matrix and you say that the data comes from the valid_ratings at the same time
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

# the nnz command counts the number of non zero elements in that specific matrix or column.
print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
return valid_ratings, train, test
```
