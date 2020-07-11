# Optimization for Machine Learning

In this repository we have the exercise notes for the course on the optimization for machine learning.Notebooks are browser based, and you start it on your localhost by typing jupyter notebook in the console. This will launch a new browser window (or a new tab) showing the Notebook Dashboard, a sort of control panel that allows (among other things) to select which notebook to open. For the moment it only contains the practical exercises since I don't know if I will take this course next year. 

You should always start the notebook with the following commands :

```
# Plot figures in the notebook (instead of a new window)
%matplotlib notebook

# Automatically reload modules
%load_ext autoreload
%autoreload 2

# The usual imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

The command `a` adds a cell above the commands we have and the command `b` adds them below. Then you can check the python version as follows :

```
# Check the Python version
import sys
# meaning that you are analyzing the version numbers 
if sys.version.startswith("3."):
  print("You are running Python 3. Good job :)")
else:
  # so the printing will be done in the console. 
  print("This notebook requires Python 3.\nIf you are using Google Colab, go to Runtime > Change runtime type and choose Python 3.")
```

Matplotlib is used for plotting, plots are directly embedded in the notebook thanks to the '%matplolib inline' command at the beginning. We have that `np.random.randn(d0, d1...)` returns a sample from the normal distribution where the d0, d1 etc parameters are the dimensions of the array that we are returning. We have that all numpy operations applying on an array can be called np.function(a) or a.function() (i.e np.sum(a) or a.sum()). You can cast an integer as a float. To define a prederfined array you can write `a = np.array([[1.0, 2.0], [5.0, 4.0]])`. When you write `b = a` this is only a referential copy and if you want a deep copy you need to write `b = a.copy()`.

The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. NumPy’s broadcasting rule relaxes this constraint when the arrays’ shapes meet certain constraints. The simplest broadcasting example occurs when an array and a scalar value are combined in an operation. When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions and works its way forward. Two dimensions are compatible when they are equal, or one of them is 1. Ravel creates a flattened view of an array (1-D representation) whereas flatten creates flattened copy of the array. Reshape allows in-place modification of the shape of the data. Transpose shuffles the dimensions.
And np.newaxis allows the creation of empty dimensions.

You can write the tranpose of a matrix in several ways as follows : `a.T, a.tranpose(), np.transpose(a)`. To add a new axis to an array c you can write the following : `c[np.newaxis]`. ou can print the shape of the array as follows : `print(c.shape)`. Reduction operations (np.sum, np.max, np.min, np.std) work on the flattened ndarray by default. You can specify the reduction axis as an argument. Meaning that first you need to flatten the array, and then you can apply on this the operations. So maybe the following really are equivalent ?

```
np.sum(a), np.sum(a, axis=0), np.sum(a, axis=1) # reduce-operations reduce the whole array if no axis is specified
```

The numpy.allclose method returns true if the two parameters on either side are equal to a certain tolerance. For other linear algebra operations, use the np.linalg module `np.linalg.eig(a)`. Mext we will find the inverse of matrix a as follows : `np.linalg.inv(a)`. In the following example for example we have that :

```
np.allclose(np.linalg.inv(a) @ a, np.identity(a.shape[1]))
```

The first parameter is the inverse of matrix a times a and then we compare that to the identity matrix that is a square matrix of size the number of columns twice. We can also solve systems of linear equations : `np.linalg.solve(a, v)` which solves the system `ax = v`. I guess that the following means that we we are adding a new column `v[:,np.newaxis]`. The following probably creates an array of size 3x4 with random integers from 0 to 9. 

```
r = np.random.random_integers(0, 9, size=(3, 4))
```

You can use the following syntax to select the columns 1 through to 3 : `r[:, 1:3]`. Using logical operations on arrays give a binary mask. Using a binary mask as indexing acts as a filter and outputs just the very elements where the value is True. This gives a memoryview of the array that can get modified. An example is : `r[r > 5]`. You can also work on subparts of an array by directly working on indices as follows : `np.where(r == 999)`. This gives the indices where the condition is true, it gives a tuple whose length, is the number of dimensions of the input array. You can combine all this and even write for example : 

```
# returns the indices where the vector has entries smaller than 5 strictly 
print(np.where(np.arange(10) < 5))
# and then I suppose we are finding the first index where the condition above is satisfied
np.where(np.arange(10) < 5)[0]
```

We also have the following ternary condition formulation : `np.where(r == 999, -10, r+1000)`. Here for example if indeed r is equal to 999, then we take -10 as the result, otherwise we take r+1000 as the result. To get the view corresponding to the indices you can write : `r[(np.array([1,2]), np.array([2,2]))]`. Here you have the subarray with the given indices as specified. You can avoid writing for loops as follows : `%timeit np.sum(numbers > 0)`. Here you select the elements in the vector which are strictly positive, and then you take the sum of the elements and you can output this. The following too for example `%timeit 1 + X + X**2 + X**3 + X**4`, which in this case applies the given operation to all the respective coordinates of that vector. In this case we have `X = np.random.randn(10000)`.

With SciPy you can do some plotting functions as follows : `from scipy.fftpack import fft, plt.plot(fft(X).real)`.
