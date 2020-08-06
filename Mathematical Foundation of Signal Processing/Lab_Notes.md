 # Notes from the Lab
 
 **Lab 1**
 
 First we import the necessary packages as follows :
 
 ```
 # %pylab inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import signal
from scipy.fftpack import dct
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.io import imread
from tqdm import tqdm_notebook as tqdm

%matplotlib inline
```

We have the following example with the ECG signal :

```
# must be the generate from text method associated to the numpy library
# the data is in a .dat file and the data is all separated with a comma 
data_ecg = np.genfromtxt('ecg.dat', delimiter=',')
x = data_ecg[:-1, 1] 
t = data_ecg[:-1, 0]
# the length of the vector 
N = len(x)
# plotting one vetor against the other 
plt.plot(t, x)

# create a box function
box = np.zeros(N)
# here int rounds the result down to the nearest integer, so you can select this way the index at the given row and column number and set
# that element equal to one on 41.
box[int(N/2) - 20:int(N/2) + 21] = 1/41
plt.plot(box)
plt.title('Box function')
```

We do the convolving as follows :

```
# here we are creating a matrix of zeroes which is N by N 
A = np.zeros((N, N))
for i in range(N):
    # Where here we are taking the ith row in its entirety 
    # numpy.roll(array, shift, axis = None) : Roll array elements along the 
    # specified axis. Basically what happens is that elements of the input array are 
    # being shifted. If an element is being rolled first to last-position, it is rolled back to first-position.
    A[i:]=np.roll(box, int(i - N / 2))

# we are printing the shape of the matrix 
print(A.shape)
# the parameters of the figure size are given 
plt.figure(figsize=(16, 4))
plt.imshow(A+0.001, interpolation='None', norm=LogNorm(vmin=0.001, vmax=10))
```

We calculate the convolution as follows :

```
y = A @ x
plt.plot(y)
plt.title('Blurred signal')

# A is invertible so we can use the inverse of A to recover x
x_hat = np.linalg.inv(A) @ y
```
We plot the result and check that it is close to x :

```
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
ax[0].plot(t, x)
ax[1].plot(t, x_hat)
ax[0].set_title('Original (x)')
ax[1].set_title('Estimate (x_hat)')
plt.show()
print('Mean squared error (MSE) is', np.mean((x-x_hat)**2))
```
Where here we are considering the mean of the difference of the two vectors where you take all the elements to the power of two. We verify that the estimator is correct as follows :

```
# Since we used a right inverse, we should have consistency: let's check
plt.plot(t2,A2 @ x)
plt.plot(t2,A2 @ x_hat)
```
Below we use the discrete cosine transform matrix :

```
# construct matrix B so R(B) is M lowest frequencies 
# the shape returns two parameters for the number of rows and columns of the matrix 
[M, N] = A2.shape
B = dct(np.eye(N)) #create an NxN DCT matrix
# we select all the rows and then all the columns until the M+1 column
B = B[:, :M] 
```
