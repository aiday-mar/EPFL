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

Estimate x by projecting onto the range of \beta while maintaining consistency. We have the following code :

```
x_hat = B @ np.linalg.inv(A2 @ B) @ y2
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
ax[0].plot(t, x)
ax[1].plot(t, np.real(x_hat))
ax[0].set_title('Original')
ax[1].set_title('Oblique projection onto R(B) - Consistent')
plt.show()
print('MSE is', np.mean((x-x_hat)**2))
```

We have the following code :

```
# we will find both the solution in R(B) that is closest to the affine subspace of consistent solutions
# and the solution in the affine subspace of consistent solutions that is closest to R(B)

freq_to_keep = 120
B = dct(np.eye(N))
B = B[:, :freq_to_keep] #remove all cols after freq_to_keep to only keep the freq_to_keep lowest frequencies
# where here basically we select all the rows but only select the first columns up to the column frq_to_keep

U, s, Vh = np.linalg.svd(A2) #take the SVD of A2 so that we can abstract a bases for its null space
basesNullspaceA = Vh[len(s):, :].T #abstract the null space
# where we select only the row starting from the row given by index len(s) and all the columns 
T = np.hstack([B, -basesNullspaceA]) #concatenate a bases for B with a bases for the null space of A

coeffs = np.linalg.inv(T.T@T) @ T.T @ A2.T @ np.linalg.inv(A2@A2.T) @ y2 
# solve the least squares problem (first 2*half_len coeffs are for B and the rest for the null space of A)
# here we take the transpose of matrix T, then we take the matrix product with the matrix T
# below we take the columns from the beginning to the column freq_to_keep
x_hat = B @ coeffs[:freq_to_keep] 
# point in R(B) that is closest to affine subspace of consistent solutions
x_hat2 = basesNullspaceA @ coeffs[freq_to_keep:] + A2.T @ np.linalg.inv(A2 @ A2.T) @ y2 
#consistent solution closest to R(B)
```

Now we plot both estimators :

```
fig, ax = plt.subplots(3, 2, figsize = (20, 15))
# since there are 3 by 2 subplots this means that we have three rows and two columns of different subplots
ax[0][0].plot(t, x)
ax[0][1].plot(t2, y2)
ax[0][0].set_title('Original (x)')
ax[0][1].set_title('Downsampled and blurred (y)')
ax[1][0].plot(t, x_hat)
ax[1][1].plot(t, np.real(x_hat2))
ax[1][0].set_title('Point in R(B) that is closest to affine subspace of consistent solutions')
ax[1][1].set_title('Point in affine subspace of consistent solutions that is closest to R(B)')
ax[2][0].plot(t, np.real(x_hat)-x)
ax[2][1].plot(t, np.real(x_hat2)-x)
ax[2][0].set_title('Error for point in R(B) that is closest to affine subspace of consistent solutions')
ax[2][1].set_title('Error for point in affine subspace of consistent solutions that is closest to R(B)')
plt.show()
print('MSE for point in R(B) that is closest to affine subspace of consistent solutions is', np.mean((x-x_hat)**2))
print('MSE for point in affine subspace of consistent solutions that is closest to R(B) is', np.mean((x-x_hat2)**2))
```

Now let's look at a 2D version of tomography :

```
N = 64
# Read an image from a file into an array.
image = imread(data_dir + "/phantom.png", as_gray=True)
image = rescale(image, (N / image.shape[0], N / image.shape[1]), mode='constant', multichannel = False)

plt.imshow(image, interpolation='None')
plt.gray()
```
As explained in class, in X-ray tomography, x-rays are fired through the object at different angles and the transmission is measured at the other side. To simulate these measurements, we want to be able to compute integrals at different angles. For example, it is very easy to do this horizontally and vertically by just summing the pixels.

```
# lets sum the columns to give the projection for x-rays fired vertically
# and sum the rows to give the projection for x-rays fired horizontally
fig, ax = plt.subplots(1, 2, figsize = (20, 5))
ax[0].plot(np.sum(image, 0))
ax[1].plot(np.sum(image, 1))
ax[0].set_title('Sum of columns')
ax[1].set_title('Sum of rows')

# Lets vectorise the image into a vector x
x = image.reshape(N*N)
print(x.shape)
```

Where we print the shape of the vector into the console. We then have the following code :

```
A = []
for col in range(N):
    # where we have an N x N matrix of zeroes 
    mask = np.zeros([N, N])
    # where we have all the rows and essentially column col, but why do we assign it the value 1 ?
    mask[:, col] = 1
    A.append(mask.reshape(N*N))
# meaning that now this python array is simply converted to a numpy array 
A = np.array(A)

# Let visualise a few rows from A (change the value of row and check things make sense)
print('The dimensions of A are',A.shape)
row = 10
# where below we have the given row and then all the columns
plt.imshow(A[row, :].reshape(N, N))

# And we can recalculate the sum of the columns using A (we should get the same as we did before)
plt.plot(A@x)
```

Now we add rows to the bottom of matrix A to sum the rows.

```
# where here we convert the matrix to a list ?
A = A.tolist()
for row in range(N):
    # here we have a matrix of zeroes of the given size n by n
    mask = np.zeros([N, N])
    # where here we select the whole of the row'th row
    mask[row, :] = 1
    A.append(mask.reshape(N*N))
A = np.array(A)

# We can now visualise any of the rows of the larger A
print('The dimensions of A are', A.shape)
row = 70
# here we are reshaping that specific row of the matrix 
plt.imshow(A[row, :].reshape(N, N))

def calcTomographyForwardOperator(numAngles, N):
    # between 0 and 180 we choose numAngles angles 
    theta = np.linspace(0, 180, numAngles, endpoint = False)
    A = []
    E = np.zeros((N, N))
    for i_y in tqdm(range(N)):
        for i_x in range(N):
            # where we select the element at the coordinates given by the square brackets 
            E[i_y, i_x] = 1
            R_E = radon(E, theta=theta, circle=False)
            E[i_y, i_x] = 0
            # the numpy module provides a function called numpy.ndarray.flatten()
            # which returns a copy of the array in one dimensional rather than in 
            # 2-D or a multi-dimensional array.
            A.append(R_E.flatten())
            
    # the T at the end allows us to transpose the matrix
    return np.array(A).T
    
# visualise a row
print(A.shape)
row = 505
plt.imshow(A[row,:].reshape(N, N))

# lets calculate our measurements
y = A @ x

# approximate x with A^T y
x_hat = A.T @ y
plt.imshow(x_hat.reshape(N, N))
```

Estimate x using a right inverse (or an approximation of one): you may need to remove small singular values. Plot the resulting image.

```
# calculate and plot singular values, where here we use the singular value decomposition
U, s, Vh = np.linalg.svd(A)
plt.plot(s)
plt.title('Singular values')

# trim small values, where the absolute value of the coordinate must be bigger than a certain threshold
trim_len = sum(abs(s) > 1e-2)
plt.plot(s)
# The Axes.axvline() function in axes module of matplotlib library is used to add
# a vertical line across the axis.
plt.axvline(x=trim_len)

# calculate a right inverse
s_trimmed = s[:trim_len]
# where we select up to column trim_len
U_trimmed = U[:, :trim_len]
# where we select up to row trim_len
Vh_trimmed = Vh[:trim_len, :]

# where the element in the brackets is the value which will appear on the diagonal 
right_inv = Vh_trimmed.T @ np.diag(1/s_trimmed) @ U_trimmed.T

x_hat = right_inv @ y
plt.imshow(x_hat.reshape(N, N))

# Mean squared error (MSE) between y and A@x_hat, where here you take the difference
# between the two vectors, then you take all the components to the power of two
# then you average all the entries 
print(np.mean((y - A@x_hat)**2))
```

Use a left-inverse (or an approximation of one) to estimate x. Plot the resulting image.

```
# calculate and plot singular values
U, s, Vh = np.linalg.svd(A)
plt.plot(s)
# the minimum of the entries of vector s 
print(min(s))

# trim small values
trim_len = sum(abs(s)>1e-2)
plt.plot(s)
plt.axvline(x=trim_len)

# calculate a right inverse
s_trimmed = s[:trim_len]
U_trimmed = U[:, :trim_len]
Vh_trimmed = Vh[:trim_len, :]

left_inv = Vh_trimmed.T @ np.diag(1/s_trimmed) @ U_trimmed.T
x_hat = left_inv @ y
plt.imshow(x_hat.reshape(N, N))
```

**Lab 2**

We have the following code in the `helpers_sol.py` file. 

```
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import skimage.color as color
import skimage.transform as transform
import tqdm


def rotation_matrix(angle):
    """
    Parameters
    ----------
    angle : floatm counter-clockwise rotation angle [rad].
    Returns
    -------
    R : :py:class:`~numpy.ndarray`
        (2, 2) rotation matrix.
    """
    # the numpy array is such that there is one matrix and there are two rows, in each we have two entries 
    # here also you use the cosine and sine functions related to the numpy library
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return R

def tube_decomposition(N_detector):
    """
    Compute unique characterisation (d, xi, p, w) of detector tubes.
    Parameters
    ----------
    N_detector: int, number of detectors
    Returns
    -------
    detector : :py:class:`~numpy.ndarray'
        (N_detector, 2) detector locations.
    xi : :py:class:`~numpy.ndarray`
        (N_tube, 2) normal vector to tube. (Must lie in quadrant I or II.)
    p : :py:class:`~numpy.ndarray`
        (N_tube,) tube distance from origin. (Can be negative.)
    w : :py:class:`~numpy.ndarray`
        (N_tube,) tube width.
    """
    # Uniformly distribute detectors on ring.
    detector_angle = np.linspace(0, 2 * np.pi, N_detector, endpoint=False)
    # Join a sequence of arrays along a new axis.
    detector = np.stack((np.cos(detector_angle), np.sin(detector_angle)), axis=-1)
    # Return the indices for the upper-triangle of an (n, m) array.
    dA, dB = np.triu_indices(N_detector, k=1)
    N_tube = len(dA)

    # Normal vector to detector tube, for all detector pairs.
    # This vector is always located in quadrant I or II.
    xi = detector[dA] + detector[dB]
    # we keep the dimensions 
    xi /= linalg.norm(xi, axis=-1, keepdims=True)
    # the elements in the second column that are negative are made into positive elements
    # since the elements at those indices are multiplied by -1 
    xi[xi[:, 1] < 0] *= -1

    # Tube offset from origin such that xi*p points to the tube's mid-point.
    p = np.squeeze(xi.reshape((N_tube, 1, 2)) @ 
                   detector[dA].reshape((N_tube, 2, 1)))

    # Tube width.
    intra_detector_angle = np.mean(detector_angle[1:] - detector_angle[:-1])
    M = rotation_matrix(intra_detector_angle)
    intra_detector = np.dot(detector, M.T)

    diff_vector = intra_detector[dA] - intra_detector[dA - 1]
    w = np.squeeze(diff_vector.reshape((N_tube, 1, 2)) @ 
                   xi.reshape((N_tube, 2, 1)))
    w = np.abs(w) 

    # `w` can be very close to 0 and cause problem later on. 
    # We discard these tubes for practical purposes.
    mask = ~np.isclose(w, 0)
    xi, p, w = xi[mask], p[mask], w[mask]

    return detector, xi, p, w


def sampling_op(xi, p, w, N_height, N_width, window_profile='raised-cosine'):
    """
    Numerical approximation of continuous-domain sampling operator.
    
    Parameters
    ----------
    xi : :py:class:`~numpy.ndarray`
        (N_tube, 2) normal vector to tube. (Must lie in quadrant I or II.)
    p : :py:class:`~numpy.ndarray`
        (N_tube,) tube distance from origin. (Can be negative.)
    w : :py:class:`~numpy.ndarray`
        (N_tube,) tube width.
    N_height : int
        Number of uniform vertical spatial samples in [-1, 1].
    N_width : int
        Number of uniform horizontal spatial samples in [-1, 1].
    window_profile : str
        Shape of the window. 
        Must be one of ['raised-cosine', 'rect', 'tri']
    Returns
    -------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_height*N_width) sampling operator, where each row contains the basis function of
        the instrument (vectorized row-by-row).
    """
    ### Generate grid
    Y, X = np.meshgrid(np.linspace(-1, 1, N_height), 
                       np.linspace(-1, 1, N_width), indexing='ij')
    V = np.stack((X, Y), axis=-1).reshape((N_height * N_width, 2))
    
    # We only want a regular grid on the circumcircle, hence we throw away
    # all vectors that lie outside the unit circle.
    V[linalg.norm(V, axis=-1) >= 1] = 0

    def window(x):
        """
        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (N_sample,) evaluation points.
        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            (N_sample,) values of window at `x`.
        """
        y = np.zeros_like(x, dtype=float)
        
        if window_profile == 'rect':
            # rect(x) = 1 if (-0.5 <= x <= 0.5)
            mask = (-0.5 <= x) & (x <= 0.5)
            y[mask] = 1
        elif window_profile == 'tri':
            # tri(x) = 1 - abs(x) if (-1 <= x <= 1)
            mask = (-1 <= x) & (x <= 1)
            y[mask] = 1 - np.abs(x[mask])
        elif window_profile == 'raised-cosine':
            # rcos(x) = cos(0.5 * \pi * x) if (-1 <= x <= 1)
            mask = (-1 <= x) & (x <= 1)
            y[mask] = np.cos(0.5 * np.pi * x[mask])            
        else:
            raise ValueError('Parameter[window_profile] is not recognized.')

        return y


    N_tube = len(xi)
    S = sparse.lil_matrix((N_tube, N_height * N_width), dtype=float)
    for i in tqdm.tqdm(np.arange(N_tube)):
        projection = V @ xi[i]
        x = (projection - p[i]) * (2 / w[i])
        mask = ((np.abs(x) <= 1)         &  # inner/outer boundary
                ~np.isclose(projection, 0))    # circular boundary
        S[i, mask.nonzero()] = window(x[mask])
        
    S = S.tocsr(copy=True)
    return S


def draw_tubes(S, N_height, N_width, idx, ax):
    """
    Draw detectors and detector tubes.
    
    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N_height : int
        Number of uniform vertical spatial samples in [-1, 1].
    N_width : int
        Number of uniform horizontal spatial samples in [-1, 1].
    idx : :py:class:`~numpy.ndarray`
        Tube indices to plot.
    ax : :py:class:`~matplotlib.axes.Axes`
    
    Returns
    -------
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    N_tube_wanted = len(idx)

    tubes = S[idx, :]
    if sparse.issparse(tubes):
        tubes = tubes.toarray()
        
    tubes = (tubes
             .reshape((N_tube_wanted, N_height, N_width))
             .sum(axis=0))

    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False)
    ax.set_title('Detector Tubes')
    cmap = cm.gray

    ax.imshow(tubes, cmap=cmap)
    return ax


def get_intensity(path_img, N_height, N_width, pad_size, max_rate=1e6):
    """
    Parameters
    ----------
    path_img : str
        Path to RGB image (PNG format).
    N_height : int
        Number of vertical pixels the image should have at the output.
    N_width : int
        Number of horizontal pixels the image should have at the output.
    pad_size : tuple(int, int)
        Symmetric padding [px] around (vertical, horizontal) image dimensions.
    max_rate : float
        Scale factor such that all image intensities lie in [0, `max_rate`].
    Returns
    -------
    lambda_ : :py:class:`~numpy.ndarray`
        (N_height, N_width) intensity.
    """
    lambda_rgb = plt.imread(path_img).astype(float)
    lambda_ = color.rgb2gray(lambda_rgb)
    
    # We pad the image with zeros so that the mask does not touch the detector ring.
    lambda_ = np.pad(lambda_, pad_size, mode='constant')
    lambda_ = transform.resize(lambda_, (N_height, N_width), order=1, mode='constant')
    lambda_ *= max_rate / lambda_.max()  # (N_height, N_width)

    return lambda_


def sinogram(xi, p, N, ax):
    r"""
    Plot Sinogram scatterplot, with x-axis representing \angle(xi)
    
    Parameters
    ----------
    xi : :py:class:`~numpy.ndarray`
        (N_tube, 2) normal vector to tube. (Must lie in quadrant I or II.)
    p : :py:class:`~numpy.ndarray`
        (N_tube,) tube distance from origin. (Can be negative.)
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET measurements.
    ax : :py:class:`~matplotlib.axes.Axes`
    
    Returns
    -------
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    N_tube = len(xi)
    theta = np.around(np.arctan2(*xi.T), 3)
    N = N.astype(float) / N.max()

    cmap = cm.RdBu_r
    ax.scatter(theta, p, s=N*20, c=N, cmap=cmap)

    ax.set_xlabel('theta [rad]')
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_ylabel('p')
    ax.set_ylim(-1, 1)

    ax.set_title('Sinogram')
    ax.axis(aspect='equal')

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(N)
    ax.get_figure().colorbar(mappable)
    
    return ax


def plot_image(I, title=None, ax=None):
    """
    Plot a 2D mono-chromatic image.
    Parameters
    ----------
    I : :py:class:`~numpy.ndarray`
        (N_height, N_width) image.
    title : str
        Optional title to add to figure.
    ax : :py:class:`~matplotlib.axes.Axes`
        Optional axes on which to draw figure.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(I, cmap='bone')

    if title is not None:
        ax.set_title(title)

    return ax


def backprojection(S, N):
    """
    Form image via backprojection.
    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET samples
    Returns
    -------
    I : :py:class:`~numpy.sparse.csr_matrix`
        (N_px,) vectorized image
    """
    P = S.T  # (N_px, N_tube) synthesis op
    I = P @ N

    return I


def least_squares(S, N, regularize=False):
    """
    Form image via least-squares.
    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET samples
    regularize : bool
        If `True`, then drop least-significant eigenpairs from Gram matrix.
    Returns
    -------
    I : :py:class:`~numpy.sparse.csr_matrix`
        (N_px,) vectorized image
    """
    P = S.T  # (N_px, N_tube)  # sampling operator adjoint
    G = (S @ P).toarray()
    D, V = linalg.eigh(G)

    if regularize:
        # Careful, G is not easily invertible due to eigenspaces with almost-zero eigenvalues.
        # Inversion must be done as such.
        mask = np.isclose(D, 0)
        D, V = D[~mask], V[:, ~mask]
        
        # In addition, we will only keep the spectral components that account for 95% of \norm{G}{F}
        idx = np.argsort(D)[::-1]
        D, V = D[idx], V[:, idx]
        mask = np.clip(np.cumsum(D) / np.sum(D), 0, 1) <= 0.95
        D, V = D[mask], V[:, mask]

    # The division by D may blow in your face depending on the size of G. 
    # Always regularize your inversions (as [optionally] done above)!
    I = P @ (V / D) @ V.T @ N  

    return I


def kaczmarz(S, N, N_iter, permute=False, I0=None):
    """
    Form image via Kaczmarz's algorithm.
    Parameters
    ----------
    S : :py:class:`~scipy.sparse.csr_matrix`
        (N_tube, N_px) sampling operator.
    N : :py:class:`~numpy.ndarray`
        (N_tube,) PET samples
    N_iter : int
        Number of iterations to perform.
    I0 : :py:class:`~numpy.ndarray`
        (N_px,) initial point of the optimization.
        If unspecified, the initial point is set to 0.
    Returns
    -------
    I : :py:class:`~numpy.ndarray`
        (N_px,) vectorized image
    """
    N_tube, N_px = S.shape

    I = np.zeros((N_px,), dtype=float) if (I0 is None) else I0.copy()

    if permute:
        index = np.random.permutation(N_iter)
    else:
        index = np.arange(N_iter)
        
    for k in tqdm.tqdm(index):
        idx = k % N_tube
        n, s = N[idx], S[idx].toarray()[0]
        l = s @ s

        if ~np.isclose(l, 0):
            # `l` can be very small, in which case it is dangerous to do the rescale. 
            # We'll simply drop these degenerate basis vectors.
            scale = (n - s @ I) / l
            I += scale * s

    return I
```
