 # Mathematical Foundation of Signal Processing 
 
 **Homework 2**
 
 ```
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import scipy.io
import tqdm

%matplotlib inline

# Load the relevant data from the given .mat file.
data = scipy.io.loadmat('hw2_ex3.mat')

# Read the measurement vector and store it as an array b.
b = data['measurement'].squeeze()

# Beam start and beam end points, here we must be choosing the specific column vectors of the array called data
beam_start = data['beam_start']
beam_end = data['beam_end']

# Length and width of the reconstruction domain (in meters, not in pixels).
length = data['length']
width = data['width']

# Arrays containing positions of all emitters and all detectors
# NOTE: not all beams that can be measured are actually measured,
#       that's why the measured beams are defined with beam_start and beam_end
emitter_pos = data['emitter_pos']
detector_pos = data['detector_pos']

# Plot the emitters and detectors.
# here we see that we have 1 suplot out of 1
fig, ax = plt.subplots(1, 1, figsize = (10,10))
plt.scatter(emitter_pos[:,0], emitter_pos[:,1], label='emitters')
plt.scatter(detector_pos[:,0], detector_pos[:,1], label='detectors')
plt.grid()
plt.legend()
plt.title("Array of emitters and detectors")
plt.show()
```

We then generate the measurement matrix as follows :

```
def compute_beam_line_params(beam_start, beam_end):
    """
    Compute the line parameters between points in beam_start and points in beam_end.
    
    NOTE: Make sure that the line normals are between 0 and \pi.
    
    Arguments:
        beam_start: (n_beams, 2) matrix with emitter positions
        beam_end: (n_beams, 2) matrix with detector positions
    
    Returns: 
        Matrix (n_beams, 2) of (unit-norm) line normals.
        Vector (n_beams,) of line signed-distances from the origin.
    """
    beam_angles = np.arctan2(beam_end[:,1]-beam_start[:,1], beam_end[:,0]-beam_start[:,0])
    # Rotate by \pi/2 to get the angle of a beam normal (rotation by -\pi/2 would also work).
    # the below is used to write pi where you make a call to the numpy library
    beam_normals = beam_angles + np.pi / 2
    # Make corrections to have angles in the range [0,\pi], by now they are in [-\pi/2,3\pi/2].
    # the condition within the square brackets indicates what needs to be verified for us to select the appropriate coordinates
    beam_normals[beam_normals > np.pi] -= 2 * np.pi
    beam_normals[beam_normals < 0] += np.pi
    # then you apply the cosine function, and the sine function onto all the entries of the vectors below
    beam_vectors = np.stack((np.cos(beam_normals), np.sin(beam_normals)), axis=-1)
    # Distance from any point on a line is the same, so just compute using beam_start.
    # maybe that means you are computing against the last axis ? 
    beam_dist = np.sum(beam_start * beam_vectors, axis=-1)
    
    # you can return here various elements
    return beam_vectors, beam_dist

def beam_pixel_intersection_length(beam_normals, beam_dist, pix_pos, pix_r):
    """
    Compute the lenght of the intersection between a thin beam and a round (disc) pixel.
    
    Arguments:
        beam_normals: (2,) vector (\cos(\theta), \sin(\theta)) normal to the beam
        beam_dist: signed distance of the beam line to the origin
        pix_pos: (2,) vector with (x,y) coordinates of the pixel
        pix_r: radius of the pixel disc
    
    Returns:
        Length of the beam segment inside the pixel disc.
    """
    # meaning we take the absolute value of what is in between the brackets 
    dist = np.abs(np.sum(beam_normals * pix_pos, axis=-1) - beam_dist)
    # Clipping distance to pix_r to simplify length computation.
    dist = np.minimum(dist, pix_r)
    # here we must be taking the square of all the coordinates in the vector
    intersection_length = 2 * np.sqrt(pix_r**2 - dist**2)
    
    return intersection_length


def construct_forward_matrix(beam_normals, beam_dist, pixel_pos, pixel_r):
    # here we have the description in between triple quotes
    """
    Constructs the forward matrix from beam line parameters and pixel grid parameters.
    
    Arguments:
        beam_normals: (n_beams, 2) matrix with beam normal vectors (\cos(\phi_i), \sin(\phi_i) in every row
        beam_dist: (n_beams,) vector with beams' signed distances from the origin
        pixel_pos: (n_pixels, 2) matrix with pixel coordinates (x_i,y_i) in every row
        pixel_r: pixel radius
    
    Returns:
        Measurement matrix of size (n_beams, n_pixels) for the given beams
        and the given pixel grid.
    """
    # Here we're using a relatively ugly but probably more efficient way of computing coefficients
    # (a simple and more time-consuming way would use nested for loops).
    # We split the x and y coordinates of beam normals and pixel positions in MxN matrices,
    # which allow us to compute distances with simple matrix elemenet-wise products and sums.
    # where you take the shape and then you choose the number of rows by writing shape[0]
    beam_x = np.kron(np.reshape(beam_normals[:,0], (-1,1)), np.ones((1, pixel_pos.shape[0])))
    beam_y = np.kron(np.reshape(beam_normals[:,1], (-1,1)), np.ones((1, pixel_pos.shape[0])))
    pixel_x = np.kron(np.reshape(pixel_pos[:,0], (1,-1)), np.ones((beam_normals.shape[0], 1)))
    pixel_y = np.kron(np.reshape(pixel_pos[:,1], (1,-1)), np.ones((beam_normals.shape[0], 1)))
    beam_dist_mat = np.kron(np.reshape(beam_dist, (-1,1)), np.ones((1, pixel_pos.shape[0])))
    # Now the distance computation becomes simple.
    # It could probably be even simpler and more elegant with multidimensional matrices,
    # but let's leave it at this.
    beam_pixel_dist = np.abs(beam_x * pixel_x + beam_y * pixel_y - beam_dist_mat)
    # Anything above pixel radius clipped to the radius which makes the coefficient zero.
    beam_pixel_dist = np.minimum(beam_pixel_dist, pixel_r)
    forward_matrix = 2 * np.sqrt(pixel_r**2 - beam_pixel_dist**2)
    
    return forward_matrix
    

# Here we proceed by computing beam_normals and beam_dist from beam_start and beam_end
beam_normals, beam_signed_dist = compute_beam_line_params(beam_start, beam_end)

# Then we generate the pixel grid. 
# Hint 1: the number of pixels could be equal to the number of measurements
# Hint 2: set pixel sizes according to the number of pixels and the size of the reconstructed domain.
# the length is an integer value
n_pixels = len(b) 

# Pixel distances along x and y.
dx = np.sqrt(width * length / n_pixels)

# Pixel size (radius) along x and y (pixels modeled as discs).
pixel_r = dx / np.sqrt(2)

# Generate the pixel grid
pixel_grid_x = np.arange(dx/2, width, dx)
pixel_grid_y = np.arange(dx/2, length, dx)

# where here you are trying to find the length of this vector 
nx = len(pixel_grid_x)
ny = len(pixel_grid_y)

# Kronecker product of two arrays.
pixel_x = np.kron(np.reshape(pixel_grid_x, (-1,1)), np.ones((pixel_grid_y.shape[0],1)))
pixel_y = np.kron(np.ones((pixel_grid_x.shape[0],1)), np.reshape(pixel_grid_y, (-1,1)))

# join a sequence of arrays along an existing axis.
# in which axis=0 along the rows (namely, index in pandas), and axis=1 along the columns
pixel_pos = np.concatenate((pixel_x, pixel_y), axis=-1)

# Finally, we compute the forward matrix (i.e. the measurement matrix).
A = construct_forward_matrix(beam_normals, beam_signed_dist, pixel_pos, pixel_r)
```

We then have the following code for the Kaczmarz method : 

```
def kaczmarz(A, b, n_iter, limits=None, randomize=False, f_0=None):
    """
    Form image via Kaczmarz's algorithm.

    Parameters
    ----------
    A : :py:class:`~numpy.ndarray`
        (n_beams, n_pixels) measurement matrix.
    b : :py:class:`~numpy.ndarray`
        (n_beams,) vector with measurements.
    n_iter : int
        Number of iterations to perform.
    limits : :py:class:`~numpy.ndarray`
        (2,) pixel value constraints. 
        Each pixel of the output image must lie in [`limits[0]`, `limits[1]`].
        If `None`, then the range is not restricted.
    randomize: bool
        Apply a ranomization strategy when applying projections.
    f_0 : :py:class:`~numpy.ndarray`
        (n_pixels,) initial point of the optimization.
        If unspecified, the initial point is set to an all-zero vector.

    Returns
    -------
    f : :py:class:`~numpy.ndarray`
        (n_pixels,) vectorized image
    """
    
    # because the shape method returns two values, one for the number of rows and the second for the number of columns 
    n_beams, n_pixels = A.shape
    
    # how come in the following we don't have a second parameter for the shape of the matrix of zeroes ? We are also saying that the type in the matrix is float
    # also here we initialize the variable and we follow this up with an if else condition
    f = np.zeros((n_pixels,), dtype=float) if (f_0 is None) else f_0.copy()

    if randomize:
        norms_sq = np.sum(A * A, axis=-1)
        probs = norms_sq / sum(norms_sq)
        # Python random module has a function choice() to randomly choose an item from a list and other sequence types.
        index = np.random.choice(n_beams, size=n_iter, p=probs)
    else:
        # where here we are taking the modulo the number of beams 
        # Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop). 
        # For integer arguments the function is equivalent to the Python built-in range function, but returns an ndarray rather than a list.
        # when not specified we have here a step size of one 
        index = np.arange(n_iter) % n_beams
        
    for k in tqdm.tqdm(index):
        n, s = b[k], A[k,:]
        # between two matrices, it is the matrix multiplication and otherwise it symbolizes the decorator 
        # a function which can accept a function as an argument and which will return a function Decorators 
        # are used often in Python to provide modifications to the existing functionality of defined function
        l = s @ s
        
        # this is the bitwise complement unary operation in python
        # The isclose() function is used to returns a boolean array where two arrays are element-wise equal within a tolerance.
        # The tolerance values are positive, typically very small numbers. 
        # The relative difference (rtol * abs(b)) and the absolute difference atol are added together to compare against the absolute difference between a and b.
        if ~np.isclose(l, 0):
            # `l` can be very small, in which case it is dangerous to do the rescale. 
            # We'll simply drop these degenerate basis vectors.
            # so here you must be doing element wise division no ?
            scale = (n - s @ f) / l
            f += scale * s
        
        if limits:
            # Given an interval, values outside the interval are clipped to the interval edges. For example, 
            # if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
            f = np.clip(f, limits[0], limits[1])

    return f
 ```

We also have the following code in python once again :

```
# two times the number of rows in this case where the number of rows is given by the method .shape[0]
im0 = kaczmarz(A, b, 2*A.shape[0], randomize=False, limits=None).reshape(ny, nx)
im1 = kaczmarz(A, b, 2*A.shape[0], randomize=True, limits=None).reshape(ny, nx)
im2 = kaczmarz(A, b, 2*A.shape[0], randomize=True, limits=[0,1]).reshape(ny, nx)

def plot_image(I, title=None, ax=None):
    """
    Plot a 2D mono-chromatic image.

    Parameters
    ----------
    I : :py:class:`~numpy.ndarray`
        (n_height, n_width) image.
    title : str
        Optional title to add to figure.
    ax : :py:class:`~matplotlib.axes.Axes`
        Optional axes on which to draw figure.
    """
    if ax is None:
        # here we have that the first parameter sent from the subplots() method is not used 
        _, ax = plt.subplots()
        
    # Display data as an image; i.e. on a 2D regular raster.
    # cmapstr or Colormap, optional
    ax.imshow(I, cmap='bone')

    # here title is a variable holding the string value of the title that we want to set 
    if title is not None:
        ax.set_title(title)

# you have the three axes as well as the figure, where you go from subplot 1 to 3 
# you have the digure size which is given as the third parameter and it is a set of proportions
fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize = (15,10))

# you plot the different images, they have different titles and we have different corresponding axes 
plot_image(im0, title='Kaczmarz', ax=ax0)
plot_image(im1, title='randomized Kaczmarz', ax=ax1)
plot_image(im2, title='randomized Kaczmarz with box constraints', ax=ax2)
```
