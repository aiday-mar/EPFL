# Exercise Notes

**Lab 1**

`a.max(0)` finds the maximum element for each column of matrix a,
`a.max(1)` finds the maximum element for each row of matrix a,
`a.dot(b)` is the matrix multiplication of two matrices a and b,
`a.shape[k]` returns the size along dimension k of the matrix a,
`np.sum(a, axis = k)` sums the elements of matrix a along dimension k,
`linalg.inv(a)` returns the inverse of matrix a

Once Git is installed you can pull a github repository using: git clone <url.git>, e.g. git clone https://github.com/epfml/ML_course.git. You do not want to break the master branch by mistake, so you start by creating and moving to a new branch: git checkout -b <name-of-my-new-branch>. Here the -b flag stands for the word branch. Then when you are on your branch, you can change the files as if you were on the master branch :
        
```
# let's say we modify file.py here
git status # check the status of the files git is tracking
# you can individually add the necessary files into the branch you checked out
git add file.py
git commit -m "some message clearly explaining the modification"
```

You can also push to the origin as follows : `git push origin <name-of-my-new-branch>`. Once your PR is accepted and merged, do not forget to switch back to master: git checkout master and pull your approved changes git pull origin master. Often in the python interpreted code you write the following at the beginning of the line `%matplotlib inline`, as well as `%load_ext autoreload, %autoreload 2`. Plots are directly embedded in the notebook thanks to the '%matplolib inline' command at the beginning.

You can define a function as follows using 

```
x = [2,3,4]

def my_function(l):
    l.append(12)
```

Here we have the vector x, and we take this vector x into the parameters of the function and then append a scalar or 12 at the end. Consider now the following histogram method in the matplotlib library :

```
matplotlib.pyplot.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, \*, data=None, \*\*kwargs)
```

Where the first parameter x is the array or the sequences of the arrays for which we must drsw a histogram. If bins is an integer, it defines the number of equal-width bins in the range. The lower and upper range of the bins. Lower and upper outliers are ignored. If not provided, range is (x.min(), x.max()). In the above the array is defined as an array of 1000 random integers. You can define numpy arrays as follows :

```
np.array([[1,3,4],[2,5,6]])
```

Where one set of square brackets implies that there is this one row, and inside of it you have two more rows. Next `np.arange(10)` creates an array which starts from 0 and goes to 9. Consider now the following `tensor_3 = np.ones((2, 4, 2))`. Then this has one pair of square brackets. Then two rows, then 4 rows in each of the two rows, and each such row has two columns. Now the following are not methods, these are attributes `tensor_3.shape, tensor_3.dtype`. You can see these are not methods because there are no brackets after the word shape or dtype. 

Following this, it would seem that writing b = a implies that if b is changed then a is also. However if you write `b = a.copy()` then when you change b then a does not change. The next thing we learn is how to slice the arrays in python. 

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

**Problem set 11**

Let's implement basic linear regression using the PyTorch library : 

```
# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
%load_ext autoreload
%autoreload 2
```
We define the following dataset as a test data set :

```
# Defining a toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
```

We have the following code which can be used to implement the linear regression. 

```
class MyLinearRegression:
    def __init__(self):
        # TODO: Define the parameters of the model (weights and biases)
        # the w variable of the current instance self is a new variable where we have a tensor initialized with only one element,
        # namely zero, what is this requires_grad variable ?
        self.w = Variable(torch.Tensor([0]), requires_grad=True)
        self.b = Variable(torch.Tensor([0]), requires_grad=True)
    
    def forward(self, x):
        # TODO: implement forward computation - compute predictions based on the inputs
        return self.w * x + self.b
    
    def parameters(self):
        # TODO: this function should return a list of parameters of the model
        return [self.w, self.b]
    
    def __call__(self, x):
        # Convenience function, where we return the forward method evaluated in x and associated to this instance
        return self.forward(x)
    

def mselossfunc(pred, y):
    # TODO: implement the MSE loss function
    return (pred - y).pow(2).mean()

# we initialize an instance of this class, for this we do not need to use the new keyword, we just write the class name and ()
model = MyLinearRegression()
# we have an array with three elements inside of it and the second parameter is the float variable 
numpy_inputs = np.asarray([0.0, 1.0, 2.0], dtype = np.float32)
# you can convert a symple numpy array into a torch variable 
torch_inputs = Variable(torch.from_numpy(numpy_inputs))
# what does the below mean ?
torch_outputs = model(torch_inputs)
print("Testing model: an input of %s gives a prediction:\n %s" % (numpy_inputs, torch_outputs))
```

We then have the following train and visualization methods :

```
def train(features, labels, model, lossfunc, optimizer, num_epoch):

    for epoch in range(num_epoch):
        # TODO: Step 1 - create torch variables corresponding to features and labels
        inputs = Variable(torch.from_numpy(features))
        targets = Variable(torch.from_numpy(labels))

        # TODO: Step 2 - compute model predictions and loss
        # meaning you input the inputs into the model
        outputs = model(inputs)
        # where the lossfun is the 4th parameter above
        loss = lossfunc(outputs, targets)
        
        # TODO: Step 3 - do a backward pass and a gradient update step
        # we are going to take a gradient with only zero components
        optimizer.zero_grad()
        # not sure what this backward method is used for on the loss 
        loss.backward()
        # you need to update the step of the optimizer.
        optimizer.step()
        
        # meaning that it is a mulitple of 10.
        if epoch % 10 == 0:
            # we have the percent sign and d must mean decimal/digit? f must mean float
            print ('Epoch [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epoch, loss.item()))
        
        
def visualize(x_train, y_train, model):
    # A convenience function for visualizing predictions given by the model
    # here you put the torch variable into the model, you take the data by writing .data, and convert to a numpy structure with 
    # numpy() 
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    # sort against the row
    order = np.argsort(x_train, axis = 0)
    # below when you plot you need to first order the x_train vector, what is flatten used for 
    plt.plot(x_train[order].flatten(), y_train[order].flatten(), 'ro', label='Original data')
    plt.plot(x_train[order].flatten(), predicted[order].flatten(), label='Fitted line')
    plt.legend()
    plt.show()
```

Then we have :

```
# Training and visualizing predictions made by linear regression model
# here you use the sub gradient descent, with the right parameters appeared first and then you have the tolerance as second parameter
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

train(features = x_train,
      labels = y_train,
      model = model,
      lossfunc = mselossfunc, 
      optimizer = optimizer,
      num_epoch = 50)
visualize(x_train, y_train, model)
```

We do the same with the NN package, we have :

```
class NNLinearRegression(nn.Module):
    # here we have the initialization method which is applied on the current instance
    def __init__(self):
        # does super refer to the constructor of the parent class, but what would this parent class be ? 
        super(NNLinearRegression, self).__init__()
        # TODO: Define the parameters of the model (linear nn layer)
        # what is this Linear function part of the NN package, and the two parameters following it ? 
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        # TODO: implement forward computation
        return self.linear(x)
    
# Training and visualizing predictions made by linear regression model (nn package)
# TODO: use loss function from nn package
lossfunc = nn.MSELoss()

model = NNLinearRegression()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

train(features = x_train,
      labels = y_train,
      model = model,
      lossfunc = lossfunc,
      optimizer = optimizer,
      num_epoch = 100)
visualize(x_train, y_train, model)
```

We have here the machine learning algorithm :

```
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        # TODO: Define parameters / layers of a multi-layered perceptron with one hidden layer
        self.fc1 = nn.Linear(1, hidden_size)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, 1) 
    
    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return out
    
# TODO: Play with learning rate, hidden size, and optimizer type for multi-layered perceptron
hidden_size = 2
learning_rate = 1e-1

# here we are essentially placing the parameter hidden_size into the MLP class and this is used in the _init_ method 
model = MLP(hidden_size = hidden_size)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train(features = x_train,
      labels = y_train,
      model = model,
      lossfunc = lossfunc,
      optimizer = optimizer,
      num_epoch = 300)
visualize(x_train, y_train, model)
```

**Problem set 12**

Then we have :

```
# Useful starting lines
%matplotlib inline

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
```

We have :

```
def sigmoid(t):
    """apply sigmoid function on t."""
    #  we have exponential inthenumpy
    return 1.0 / (1 + np.exp(-t))

def grad_sigmoid(t):
    """return the gradient of sigmoid on t."""
    return sigmoid(t) * (1 - sigmoid(t))
```

We have :

```
# here we have 4 elements in this array
x = np.array([0.01, 0.02, 0.03, 0.04])
# below we have a dictionary where you map a word/name to a matrix of diagonal ones
W = {
    "w_1": np.ones((4, 5)),
    "w_2": np.ones(5)
}
y = 1
```

Now we have problem 1 :

```
def simple_feed_forward(x, W):
    """Do feed forward propagation.""" 
    x_0 = x
    # here we havea transpose matrix, what is this @ symbol 
    z_1 = W["w_1"].T @ x_0
    x_1 = sigmoid(z_1)
    z_2 = W["w_2"].T @ x_1
    y_hat = sigmoid(z_2)
    
    # we return three elements
    return z_1, z_2, y_hat

try:
    expected = 0.93244675427215695
    # meaning we don't really need the third return value
    _, _, yours = simple_feed_forward(x, W)
    # must be that you take the square of all the respective components and separately
    # sum the components. We assert a phrase and this returns a boolean.
    assert np.sum((yours - expected) ** 2) < 1e-15
    print("Your implementation is correct!")
except:
    print("Your implementation is not correct.")
```
Now we consider problem two. We consider the backpropagation in neural networks. We have :

```
def simple_backpropagation(y, x, W):
    """Do backpropagation and get delta_W."""
    # Feed forward, and we get three return values 
    z_1, z_2, y_hat = simple_feed_forward(x, W)
    x_1 = sigmoid(z_1)
    # Backpropogation
    delta_2 = (y_hat - y) * grad_sigmoid(z_2)
    delta_w_2 = delta_2 * x_1
    delta_1 = delta_2 * W["w_2"] * grad_sigmoid(z_1)
    delta_w_1 = np.outer(x, delta_1)
    
    # we have an example of the outer product
    # import numpy
    # x = numpy.array([1, 2, 3])
    # y = numpy.array([4, 5, 6])
    # x.__class__ and y.__class__ are both 'numpy.ndarray'

    # outer_product = numpy.outer(x, y)
    # outer_product has the value:
    # array([[ 4,  5,  6],
    #        [ 8, 10, 12],
    #        [12, 15, 18]])

    return {
        "w_2": delta_w_2,
        "w_1": delta_w_1
    }
  
try:
    # now we have an array where each row is between square brackets
    expected = {
        'w_1': np.array([
            [ -1.06113639e-05,  -1.06113639e-05,  -1.06113639e-05, -1.06113639e-05,  -1.06113639e-05],
            [ -2.12227277e-05,  -2.12227277e-05,  -2.12227277e-05, -2.12227277e-05,  -2.12227277e-05],
            [ -3.18340916e-05,  -3.18340916e-05,  -3.18340916e-05, -3.18340916e-05,  -3.18340916e-05],
            [ -4.24454555e-05,  -4.24454555e-05,  -4.24454555e-05, -4.24454555e-05,  -4.24454555e-05]]),
        'w_2': np.array(
            [-0.00223387, -0.00223387, -0.00223387, -0.00223387, -0.00223387])
    }
    yours = simple_backpropagation(y, x, W)
    # we take the square of each component and then take the sum of all the components, where we iterate over the keys
    assert np.sum([np.sum((yours[key] - expected[key]) ** 2) for key in expected.keys()]) < 1e-15
    print("Your implementation is correct!")
except:
    print("Your implementation is not correct!")
```

Now we are considering problem three. For which the answer is a theoretical one on paper.

**Problem set 13**

First we will want to see how classifier models are sensitive to noise. For this we will need to first import the right libraries.

```
%matplotlib inline
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
```

We evaluate the classifier by looking at its accuracy on the test set :

```
def accuracy(predicted_logits, reference):
    """
    Compute the ratio of correctly predicted labels
    
    @param predicted_logits: float32 tensor of shape (batch size, num classes)
    @param reference: int64 tensor of shape (batch_size) with the class number
    """
    
    labels = torch.argmax(predicted_logits, 1)
    # checking the equality of the reference to the given computed max label 
    correct_predictions = labels.eq(reference)
    # that last method probably is a measure of the number of elements nelement()
    return correct_predictions.sum().float() / correct_predictions.nelement()
  
def test_accuracy():
    # the tensor is initialized in such a way that a row is specified within one pair of square brackets and the whole matrix
    # is specified within another pair of square brackets 
    predictions = torch.tensor([[0.5, 1.0, 0.7], [-0.6, -0.3, 0]])
    correct_labels = torch.tensor([0, 2])  # first is wrong, second is correct
    # allclose method :
    # Returns True if two arrays are element-wise equal within a tolerance.
    assert accuracy(predictions, correct_labels).allclose(torch.tensor([0.5]))

    predictions = torch.tensor([[0.5, 1.0, 0.7], [-0.6, -0.3, 0], [-1, 0, 1]])
    correct_labels = torch.tensor([1, 1, 2])  # correct, wrong, correct
    assert accuracy(predictions, correct_labels).allclose(torch.tensor([2/3]))

    print("Tests passed")
  
test_accuracy()
```
We have the following class used to perform the logistic regression:

```
class LogisticRegressionModel(torch.nn.Module):
    # the method that will be called when you initialize the class
    def __init__(self):
        # wha is the super parent class in this case ? 
        super().__init__()
        self.image_area = 28 * 28 # pixels
        self.num_classes = 10
        # what is this Linear function which is mentioned ? Part of the neural network package
        self.linear_transform = torch.nn.Linear(self.image_area, self.num_classes, bias=True)

    def forward(self, x):
        # the batch size corresponds to the number of rows of x
        batch_size = x.shape[0]
        # the view method akin to the reshape method allows to change the size of the image 
        flattened_images = x.view(batch_size, self.image_area)
        return self.linear_transform(flattened_images)
```

In the below we train the logistic regression model : 

```
def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs):
  # here in between the triple quotes we have the @param, the name of the parameter, :, and the type 
  """
  @param model: torch.nn.Module
  @param criterion: torch.nn.modules.loss._Loss
  @param dataset_train: torch.utils.data.DataLoader
  @param dataset_test: torch.utils.data.DataLoader
  @param optimizer: torch.optim.Optimizer
  @param num_epochs: int
  """
  print("Starting training")
  for epoch in range(num_epochs):
    # Train an epoch
    model.train()
    for batch_x, batch_y in dataset_train:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)

      # Evaluate the network (forward pass), where you use the x coordinates now to make a prediction
      prediction = model(batch_x)
      loss = criterion(prediction, batch_y)
      
      # Compute the gradient
      optimizer.zero_grad()
      loss.backward()

      # Update the parameters of the model with a gradient step
      optimizer.step()

    # Test the quality on the test set
    model.eval()
    accuracies_test = []
    for batch_x, batch_y in dataset_test:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)

      # Evaluate the network (forward pass)
      prediction = model(batch_x)
      accuracies_test.append(accuracy(prediction, batch_y))
    
    # you print the epoch, and you take the average of the accuracies where you are using here a float type
    print("Epoch {} | Test accuracy: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))
```
Then we have :

```
num_epochs = 10
learning_rate = 1e-3
batch_size = 1000

dataset_test = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data', train=False, download=True, transform=torchvision.transforms.ToTensor()), 
  batch_size=100,
  shuffle=True
)
dataset_train = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data', train=True, download=True, transform=torchvision.transforms.ToTensor()),
  batch_size=batch_size,
  shuffle=True
)

# If a GPU is available (should be on Colab, we will use it)
if not torch.cuda.is_available():
  # this exception has a string inside 
  raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the logistic regression model with the Adam optimizer
criterion = torch.nn.CrossEntropyLoss() # this includes LogSoftmax which executes a logistic transformation
model_logreg = LogisticRegressionModel().to(device)

optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)
train(model_logreg, criterion, dataset_train, dataset_test, optimizer, num_epochs)

# You should expect a test accuracy of around 91%.
# Training should take around a minute
```
We use the tools now to build a simple convolutional network. 

```
class LeNetModel(torch.nn.Module):
  def __init__(self):
    """From: LeCun et al., 1998. Gradient-Based Learning Applied to Document Recognition"""
    super().__init__()
    # Applies a 2D convolution over an input signal composed of several input planes.
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
    # The term "dropout" is used for a technique which drops out some nodes of the network. Dropping out can be seen as temporarily
    # deactivating or ignoring neurons of the network. This technique is applied in the training phase to reduce overfitting effects.
    # Overfitting is an error which occurs when a network is too closely fit to a limited set of input samples.

    self.conv2_drop = torch.nn.Dropout2d()
    self.fc1 = torch.nn.Linear(320, 50)
    # CLASS torch.nn.Linear(in_features, out_features, bias=True)
    # Applies a linear transformation to the incoming data: y = x*W^T + b
    # Parameters:
    # in_features – size of each input sample (i.e. size of x)
    # out_features – size of each output sample (i.e. size of y)
    # bias – If set to False, the layer will not learn an additive bias. Default: True

    self.fc2 = torch.nn.Linear(50, 10)

  def forward(self, x):
    relu = torch.nn.functional.relu
    # A Tensor of format specified by data_format. The max pooled output tensor. 
    max_pool2d = torch.nn.functional.max_pool2d
     
    x = relu(max_pool2d(self.conv1(x), 2))
    x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = relu(self.fc1(x))
    x = torch.nn.functional.dropout(x, training=self.training)
    x = self.fc2(x)
    return torch.nn.functional.log_softmax(x, dim=1)

model_lenet = LeNetModel().to(device)
optimizer = torch.optim.Adam(model_lenet.parameters(), lr=learning_rate)

train(model_lenet, criterion, dataset_train, dataset_test, optimizer, num_epochs)

# Expect roughly 97% accuracy on the test set.
# Training should take around two minutes
```

We update the attack as follows :

```
def fgsm_update(image, data_grad, update_max_norm):
    """
    Compute the FGSM update on an image (or a batch of images)

    @param image: float32 tensor of shape (batch_size, rgb, height, width)
    @param data_grad: float32 tensor of the same shape as `image`. Gradient of the loss with respect to `image`.
    @param update_max_norm: float, the maximum permitted difference between `image` and the output of this function measured in L_inf norm.

    @returns a perturbed version of `image` with the same shape
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + update_max_norm*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
```

Next we decide to evaluate the attacks :

```
def evaluate_attack(model, criterion, test_loader, update_max_norm):
  """
  @param model: torch.nn.Module
  @param criterion: torch.nn.modules.loss._Loss
  @param test_loader: torch.util.data.DataLoader
  @param update_max_norm: float indicating the maximum L_infinity norm allowed for the perturbations

  @return (
    accuracy of the model in the perturbed test set,
    adversarial example images: list of 5 samples of a tuple (original prediction - float, prediction - float, example image - torch.tensor)
  )
  """
  accuracy_per_batch = []
  adversarial_examples = []  # a small sample of 5 adversarial images

  # Loop over all examples in test set in batches
  for data, target in test_loader:
    data, target = data.to(device), target.to(device)

    # Indicate that we want PyTorch to compute a gradient with respect to the
    # input batch. We have a boolean.
    data.requires_grad = True

    # Forward pass
    output = model(data)
    original_predictions = output.argmax(1) # get the index of the max logit
    original_accuracy = accuracy(output, target)
    loss = criterion(output, target)

    # Zero all existing gradients
    model.zero_grad()
    loss.backward()

    # Perturb the batch with a gradient step (using the `fgsm_update`)
    perturbed_data = fgsm_update(data, data.grad, update_max_norm)

    # Re-classify the perturbed batch
    output = model(perturbed_data)
    adversarial_predictions = output.argmax(1)
    adversarial_accuracy = accuracy(output, target)

    accuracy_per_batch.append(adversarial_accuracy)

    # Save some adversarial examples for visualization
    if len(adversarial_examples) < 5:
      adv_ex = perturbed_data[0, 0, :, :].detach().cpu().numpy()
      adversarial_examples.append( (original_predictions[0].item(), adversarial_predictions[0].item(), adv_ex) )

  average_accuracy = sum(accuracy_per_batch) / len(accuracy_per_batch)  # assuming all batches are the same size

  print("Epsilon: {:.2f}\tTest Accuracy = {:.3f}".format(update_max_norm, average_accuracy))

  return average_accuracy, adversarial_examples
```

Here tensor.detach() creates a tensor that shares storage with tensor that does not require grad. It detaches the output from the computational graph. So no gradient will be backpropagated along this variable. The wrapper with torch.no_grad() temporarily set all the requires_grad flag to false. torch.no_grad says that no operation should build the graph. The difference is that one refers to only a given variable on which it is called. The other affects all operations taking place within the with statement. Also, torch.no_grad will use less memory because it knows from the beginning that no gradients are needed so it doesn’t need to keep intermediary results.

Let's explore how the maximum allowed perturbation influences the model's test error.

```
accuracies_logreg = []
examples_logreg = []

epsilons_logreg = [0, .05, .1, .15, .2, .25, .3]

# Run test for each epsilon
for eps in epsilons_logreg:
    acc, ex = evaluate_attack(model_logreg, criterion, dataset_test, eps)
    accuracies_logreg.append(acc)
    examples_logreg.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons_logreg, accuracies_logreg, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
```

We consider the relationship directly with images :

```
# Plot several examples of adversarial samples at each epsilon
counter = 0
plt.figure(figsize=(8,10))
for epsilon, examples in zip(epsilons_logreg, examples_logreg):
    for column_number, example in enumerate(examples):
        counter += 1
        original_prediction, adversarial_prediction, img = example
        img = img.squeeze()

        plt.subplot(len(epsilons_logreg), len(examples), counter)

        plt.title("{} -> {}".format(original_prediction, adversarial_prediction))
        plt.imshow(img, cmap="gray")

        # Clear the axes
        plt.xticks([], [])
        plt.yticks([], [])

        # Print y-labels in the first column
        if column_number == 0:
            plt.ylabel("Eps: {:.2f}".format(epsilon), fontsize=14)

plt.tight_layout()
# The tight_layout() function in pyplot module of matplotlib library is used to automatically adjust subplot parameters to give 
# specified padding.
plt.show()
```

We have below an example of the use of xticks and yticks :

```
# return locs, labels where locs is an array of tick locations and
# labels is an array of tick labels.
locs, labels = xticks()

# set the locations of the xticks
xticks( arange(6) )

# set the locations and labels of the xticks
xticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )
```

We now do the same code but using convolutional neural networks :

```
accuracies_lenet = []
examples_lenet = []

epsilons_lenet = [0, .05, .1, .15, .2, .25, .3]

# Run test for each epsilon
for eps in epsilons_lenet:
    acc, ex = evaluate_attack(model_lenet, criterion, dataset_test, eps)
    accuracies_lenet.append(acc)
    examples_lenet.append(ex)

# Comparing the models below 
plt.figure(figsize=(5,5))
plt.plot(epsilons_logreg, accuracies_logreg, "*-", c='red', label='Logistic regression')
plt.plot(epsilons_lenet, accuracies_lenet, "*-", c='blue', label='Convolutional network')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))

plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend();
```

Next we will train a robust neural network model.

```
robust_neural_net_model = LeNetModel().to(device)

num_epochs = 10
# first we have the parameters then we have the constant of the learning rate ?
optimizer = torch.optim.Adam(robust_neural_net_model.parameters(), lr=1e-3)
for epoch in range(num_epochs):

    # Train an epoch
    robust_neural_net_model.train()
    for batch_x, batch_y in dataset_train:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass for adversarial perturbations
        batch_x.requires_grad = True
        output = robust_neural_net_model(batch_x)
        original_predictions = output.argmax(1) # get the index of the max logit
        original_accuracy = accuracy(output, batch_y)
        loss = criterion(output, batch_y)
        robust_neural_net_model.zero_grad()
        loss.backward()
        perturbed_data = fgsm_update(batch_x, batch_x.grad, 0.25)
        
        # Evaluate the network (forward pass)
        prediction = robust_neural_net_model(perturbed_data)
        loss = criterion(prediction, batch_y)
        
        # Compute the gradient
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters of the model with a gradient step
        optimizer.step()

    # Test the quality on the test set
    robust_neural_net_model.eval()
    accuracies = []
    for batch_x, batch_y in dataset_test:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Evaluate the network (forward pass)
        prediction = robust_neural_net_model(batch_x)
        accuracies.append(accuracy(prediction, batch_y))
      
    print("Epoch {:.2f} | Test accuracy: {:.5f}".format(epoch, sum(accuracies).item()/len(accuracies)))


  # training takes around two minutes
  # you should expect an accuracy around 96%
```

Then we have the following code :

```
accuracies_lenet_robust = []
examples_lenet_robust = []

epsilons_lenet_robust = [0, .05, .1, .15, .2, .25, .3]

# Run test for each epsilon
for eps in epsilons_lenet_robust:
    acc, ex = evaluate_attack(robust_neural_net_model, criterion, dataset_test, eps)
    accuracies_lenet_robust.append(acc)
    examples_lenet_robust.append(ex)

# Comparing the models
plt.figure(figsize=(5,5))
plt.plot(epsilons_logreg, accuracies_logreg, "*-", c='red', label='Logistic regression')
plt.plot(epsilons_lenet, accuracies_lenet, "*-", c='blue', label='Convolutional network')
plt.plot(epsilons_lenet_robust, accuracies_lenet_robust, "*-", c='orange', label='Convolutional network (robust)')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))

plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend();
```
