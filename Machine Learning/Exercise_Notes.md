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

# the order in which you return the data is important because when you call the method you will need to know this order
return valid_ratings, train, test
```
You can use the above code as follows : 

```
from plots import plot_train_test_data

valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)
plot_train_test_data(train, test)
```

We have different methods that we are implementing for the baseline method. We have :

```
from helpers import calculate_mse

def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean, by applying the mean() method on the nonzero_train vector 
    global_mean_train = nonzero_train.mean()

    # find the non zero ratings in the test, to dense probably removes the zero entries
    nonzero_test = test[test.nonzero()].todense()

    # predict the ratings as global mean
    mse = calculate_mse(nonzero_test, global_mean_train)
    rmse = np.sqrt(1.0 * mse / nonzero_test.shape[1])
    # we substitute for the variable v using the format method.
    print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))

baseline_global_mean(train, test)
```

Next we have the following code to calculate the user means :

```
def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    # where we have a vector starting from one and going all the way up to the number of elements in the num_users
    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        # where you take all the rows and you have a fixed column index
        train_ratings = train[:, user_index]
        # you first find the non zero entries of the column, then you specifically select those non zero entries
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        
        # calculate the mean if the number of elements is not 0
        # meaning that if the number of rows is not zero
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
        else:
            continue
        
        # find the non-zero ratings for each user in the test dataset
        test_ratings = test[:, user_index]
        nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nonzeros_test_ratings, user_train_mean)
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of the baseline using the user mean: {v}.".format(v=rmse))

baseline_user_mean(train, test)
```

In a similar manner we have :

```
def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
        else:
            continue
        
        # find the non-zero ratings for each movie in the test dataset
        test_ratings = test[item_index, :]
        nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nonzeros_test_ratings, item_train_mean)
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of the baseline using the item mean: {v}.".format(v=rmse))
    
baseline_item_mean(train, test)
```
Don't quite understand this initial matrix factorization below :

```
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
        
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features
```
We have the following method to compute the error :

```
def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))
```

And the main method is :

```

def matrix_factorization_SGD(train, test):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 20   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    num_epochs = 20     # number of full passes through the train set
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

matrix_factorization_SGD(train, test)
```

Implement the ALS algorithm for regularized matrix completion. For this we have the following code :

```
def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features
```

As well as the following code :

from helpers import build_index_groups

```
def ALS(train, test):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = 20   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("\nstart the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)
    print("test RMSE after running ALS: {v}.".format(v=rmse))

ALS(train, test)
```
