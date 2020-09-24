<img src="https://www.epfl.ch/about/overview/wp-content/uploads/2020/07/logo-epfl-1024x576.png" style="padding-right:10px;width:140px;float:left"></td>
<h2 style="white-space: nowrap">Image Processing Laboratory Notebooks</h2>
<hr style="clear:both">
<p style="font-size:0.85em; margin:2px; text-align:justify">
This Juypter notebook is part of a series of computer laboratories which are designed
to teach image-processing programming; they are running on the EPFL's Noto server. They are the practical complement of the theoretical lectures of the EPFL's Master course <b>Image Processing I</b> 
(<a href="https://moodle.epfl.ch/course/view.php?id=522">MICRO-511</a>) taught by Prof. M. Unser and Prof. D. Van de Ville.
</p>
<p style="font-size:0.85em; margin:2px; text-align:justify">
The project is funded by the Center for Digital Education and the School of Engineering. It is owned by the <a href="http://bigwww.epfl.ch/">Biomedical Imaging Group</a>. 
The distribution or the reproduction of the notebook is strictly prohibited without the written consent of the authors.  &copy; EPFL 2020.
</p>
<p style="font-size:0.85em; margin:0px"><b>Authors</b>: 
    <a href="mailto:pol.delaguilapla@epfl.ch">Pol del Aguila Pla</a>, 
    <a href="mailto:kay.lachler@epfl.ch">Kay Lächler</a>,
    <a href="mailto:alejandro.nogueronaramburu@epfl.ch">Alejandro Noguerón Arámburu</a>, and
    <a href="mailto:daniel.sage@epfl.ch">Daniel Sage</a>.
</p>
<hr style="clear:both">
<h1>Lab 0: Introduction</h1>
<div style="background-color:#F0F0F0;padding:4px">
    <p style="margin:4px;"><b>Released</b>: Thursday September 17, 2020</p>
    <p style="margin:4px;"><b>Submission</b>: <span style="color:red">Friday September 25, 2020</span> (before 11:59PM) on <a href="https://moodle.epfl.ch/course/view.php?id=522">Moodle</a></p>
    <p style="margin:4px;"><b>Grade weigth</b>: 1% of the overall grade</p>
    <p style="margin:4px;"><b>Remote help</b>: Monday September 21, on Zoom (see Moodle for link and time)</p>    
</div>

### Student Name: Aiday Marlen Kyzy
### SCIPER: 283505

Double-click on this cell, fill your name and SCIPER number above, and run the cell below to verify your identity in Noto and set the seed for random results.


```sos
%use sos
import getpass
# This line recovers your camipro number to mark the images with your ID
uid = int(getpass.getuser().split('-')[2]) if len(getpass.getuser().split('-')) > 2 else ord(getpass.getuser()[0])
print(f'SCIPER: {uid}')
```

    SCIPER: 283505


# <a name="top"></a> Introduction (**8 points**)

This introductory lab will guide you through the tools that you will need for the labs. As you will notice, you will be using 2 programming languages: JavaScript and Python. Don't worry if you do not have experience with them! In fact, you don't need any programming experience at all. Basic computational thinking (e.g. how *for* loops work, what a *function* is) is enough to proceed with the labs. We will always teach by example, and you can always go back and check an example we gave you to know how to program anything we ask. Moreover, you can always ask specific questions about programming to the dedicated TAs of the course.

<div class="alert alert-danger">

**Warning:** This lab is **long**. It is intended as a tutorial course for you to go back to in all the labs to come. This implies that 
<ul>
    <li> You should not worry if it feels hard, long, or a bit too much at first.
    <li> You should take your time, you have a whole week to finish it, so there's no need to finish everything in one session. While the state (variables) of the notebook will be lost, all your answers will remain. As best practice, re-run all the cells every time you restart working.
    <li> You should keep your filled copy for further reference (<b>in your local computer</b>, besides the copy that will remain in Noto). 
</ul>
</div>

At the beginning of each lab, you will always find an Index as the one below, clearly pointing at the topics that will be covered. The index will also specify how many points are awarded for each exercise, when appropriate.

## Index
1. [Introduction to Jupyter notebooks](#-1.-Introduction-to-Jupyter-notebooks) 
    1. [Cellular structure](#-1.A.-Cellular-structure)
    2. [The SoS kernel](#-1.B.-The-SoS-kernel)
2. [Python and JavaScript](#-2.-Python-and-JavaScript)
    1. [Basic Python programming](#-2.A.-Basic-Python-programming)
        1. [NumPy](#-2.A.a.-NumPy)
        2. [Matplotlib](#-2.A.b.-Matplotlib)
    2. [Basic Javascript programming](#-2.B.-Basic-JavaScript-programming)
    3. [Python vs JavaScript speed comparison](#-2.C.-Python-vs-JavaScript-speed-comparison)
3. [IPLabImageAccess](#-3.-IPLabImageAccess-(2-points)) (**2 points**)
4. [IPLabViewer](#-4.-IPLabViewer)
    1. [Creation of a viewer](#-4.A.-Creation-of-a-viewer)
    2. [Using widgets](#-4.B.-Using-widgets) (**1 point**)
    3. [User-defined widgets](#-4.C.-User-defined-widgets)
    4. [Programmatic customization](#-4.D.-Programmatic-customization)
    5. [Try it yourself!](#-4.E.-Try-it-yourself!) (**3 points**)
5. [Image processing in Python](#-5.-Image-processing-in-Python-(2-points))
    1. [Examples](#-5.A.-Examples)
    1. [Try it yourself!](#-5.B.-Try-it-yourself!-(2-points)) (**2 points**)

# <a class="anchor"></a> 1. Introduction to Jupyter notebooks
## <a class="anchor"></a> 1.A. Cellular structure

What you are looking at right now is a Jupyter Notebook. A notebook consists of cells that contain either code snippets with their results or text. This makes it the perfect place where to teach and learn image processing, as you do not need to switch between different files to see the explanation, the corresponding code, and its results. The cellular structure of Jupyter notebooks allows us to run small snippets of code one after the other, go back and forth between, or check intermediate results between them. 
<div class="alert alert-info">

**Note:** You can use the shortcut `Shift` $+$ `Enter` to run the cell and advance to the next one or `Ctrl` $+$ `Enter` to run the cell and stay in it. If you want a detailed list of keyboard shortcuts, you can check out the one in [this link](https://cheatography.com/weidadeyue/cheat-sheets/jupyter-notebook/).
</div>

It is important to know that if you declare some variable in a cell, this variable will be available in all other cells that use the same programming language, irrespective of the location of the cell. Let's look at an example: Read the description of the cells below and run them as instructed.

In this first cell we declare a variable `a` and assign it the value $5$. Run it.


```sos
%use sos
# cell 1
# this is our variable
a = 5
```

In the second cell we print the value of `a`. If you run it just after the cell above, it should print "The value of a is 5". Run it.


```sos
%use sos
# cell 2
# here we print the value of a
print(f"The value of a is {a}")
```

    The value of a is 5


Then, in the third cell, we modify `a` by adding $2$ to it. Run it.


```sos
%use sos
# cell 3
# add two to a
a = a + 2
```

Now run _cell 2_ again: the value of `a` has been modified, even though _cell 2_ is located above _cell 3_ in our notebook. Repeat this procedure (running _cell 3_ then running _cell 2_ ) and make sure you can explain the result. Now run _cell 1_ again, where we set `a=5`, and `a` will be $5$ in all cells again. Hopefully this shows the **global** impact every single cell has on the notebook. Regardless, it is good practice to design notebooks to be run from top to bottom.

<div class="alert alert-info">

**Note:** If at any point the kernel should crash for whatever reason, you can restart it by clicking on the button `Kernel` in the toolbar on top and then click on `Restart Kernel...`. Once the kernel is restarted, you will need to run all necessary cells again to regain the working environment you had before. A good practice to test that there are no errors in the notebook is to click `Restart and Run All Cells...` once you have finished a notebook. This will run all the cells from top to bottom and only stop if an error occurs.
</div>

## <a class="anchor"></a> 1.B. The SoS kernel

Surely you have noticed that the first line of all cells with code was `%use sos`. This line tells the notebook that this specific cell uses Python syntax and should be read as Python code. In normal Jupyter Notebooks, this would not be necessary, since they usually only support one single programming language in the whole notebook, defined by _the kernel_. However, for the labs, we use a special kernel that allows us to have a notebook with multiple programming languages, the [SoS](https://vatlab.github.io/sos-docs/) kernel. If you're interested you can read its fascinating [documentation](https://vatlab.github.io/sos-docs/notebook.html#Documentation). However, to understand the labs you only need to know that:
- `%use sos` indicates that a cell is written in Python,
- `%use javascript` indicates that a cell is written in JavaScript,
- `%put var_name ` converts the variable `var_name` from JavaScript to Python
- `%get var_name ` converts the variable `var_name` from Python to JavaScript

The keywords above are called _magics_ and they always need to be on the first lines of code with nothing in-between them. Let's look at an example: Read the description of the cells below and run them as instructed. Before, feel free to run the cell below to follow the instructional video by Kay Lächler, one of the student developers behind these labs. He will guide you and explain the few following cells.


```sos
from IPython.display import IFrame
IFrame(width="800", height="450", src="https://www.youtube-nocookie.com/embed/5XPVnavkgyY")
```





<iframe
    width="800"
    height="450"
    src="https://www.youtube-nocookie.com/embed/5XPVnavkgyY"
    frameborder="0"
    allowfullscreen
></iframe>




First, we declare some variable `my_var` in python.


```sos
%use sos
# This is our variable, let's initialize it to 0
my_var = 0
```

Next, we want to modify this variable in JavaScript, so we define a JavaScript cell to do that. However, we cannot simply access the variable declared in a Python cell without first converting it using one of the magics given above. Run the next cell **without modifying it**, and see if you understand the resulting error.


```sos
%use javascript
%get my_var 
%put my_var

// try to modify the python variable without converting it to JavaScript first
my_var = my_var + 5
```




    5



Now, let's do it properly. As you can see, the second and third lines are commented. These lines import the variable `my_var` from python using `%get my_var ` and convert it back to python once all the code in the cell has been executed, using `%put my_var `. Uncomment these lines (remove the  `//`) on both lines and run again.

If we now print the value of `my_var` in Python, it should be 5.


```sos
%use sos
# Print the value of the python variable my_var
print(f'The value of my_var is {my_var}')
```

    The value of my_var is 5


Note that, in the previous-to-last cell, the command `%put my_var ` only makes its effect **after** the whole cell terminates. In this case, it takes effect after the line `my_var = my_var + 5;` runs. In contrast, `%get my_var ` runs **before** the cell is run.

# <a class="anchor"></a> 2. Python and JavaScript

Now: why do we need to make everything so complicated and use two different programming languages? We can assure you it's not only to confuse you, and not only because we like the challenge. The real reason is because Python and JavaScript excel at two very distinct tasks. **_Python_ is a high-level language with many great libraries with built-in image processing algorithms, and it is great for visualizing results. On the other hand, _JavaScript_ is a fairly low-level programming language, and it is great for implementing image processing algorithms by accessing each pixel in an image.** 

The purposes of these labs are 1) for you to understand how to apply the different image processing methods to images, for which we will mostly use Python, 2) for you to learn how methods work in detail, pixel by pixel, for which we will mostly use JavaScript. 

You will usually need to implement every method **once** in JavaScript to really understand how it works inside. Then, we will show you how to apply the same method in Python by using existing image processing libraries. 

Again, do not worry, you will be provided with examples of everything that you need to do. Of course, you always have the option to contact your TAs or check [Stack Overflow](https://stackoverflow.com/), where you can find the answers to thousands of previously asked questions for any programming language.
<div class="alert alert-info">
    
**Note:** If you are already familiar with [Python](python.org) and the [NumPy](numpy.org) and [Matplotlib](matplotlib.org) libraries, you can skip to Section [2.B.](#-2.B.-Basic-JavaScript-programming).
</div>

## <a class="anchor"></a> 2.A. Basic Python programming

Declaring a variable in python is very simple. We have three basic types of variables:
```python
# Numbers
my_var = 2
my_var = 5.2
# Strings
my_var = 'Either use single quotes'
my_var = "or use double quotes"
# Boolean
my_var = True
my_var = False
```
Every variable can hold every type of data, so we don't need to specify the type at initialization.

We can have lists of several items:
```python
# List of numbers
my_list = [2.5, 7, 3.3]
# List of strings
my_list = ['E.T.', 'phone', 'home']
# List of booleans
my_list = [True, True, False]
```
Lists are 0-indexed and can be accessed with square brackets:
```python
# Set the 3rd element of my_list to 5
my_list[2] = 5
# Retrieve the 2nd element of my_list
new_var = my_list[1]
```
Similarly we can make tuples, which are just like lists, but their values cannot be modified after they have been initialized:
```python
# This is a tuple of three numbers
my_tuple = (3, 8.6, 4)
# Access to a tuple is similar to lists
new_var = my_tuple[1]
# But we cannot write to a tuple!
```
For numbers we have the following basic operators:
```python
# Define the two variables
a = 5
b = 2
# Addition
a + b
>> 7
# Subtraction
a - b
>> 3
# Multiplication
a * b
>> 10
# Division
a / b
>> 2.5
# Integer division
a // b
>> 2
# Modulo (remainder of integer division)
a % b
>> 1
# Power
a ** b
>> 25
```
We can have conditional statements
```python
if a < b:
    # printing is done with the print function
    print('a is smaller than b')
else:
    print('a is bigger than or equal to b')
```
<div class="alert alert-info">

**Note:** Indentation is important in Python. Writing a conditional statement, a loop, a function, etc., without an indent (`Tab`) raises an error.
</div>

with the following comparison operators:
```python
<  # Smaller than
<= # Smaller or equal
>  # Bigger than
>= # Bigger than or equal to
== # Equal to
!= # Not equal to
```
A `for` loop is declared as:
```python
# Note: range(n) creates numbers from 0 to n-1
for i in range(10):
    # Another for loop inside the for loop - we call this 'nested for loops'
    for j in range(10):
        # We can also have formatted strings indicated by a preceeding 'f', where the number i is inserted into the string
        print(f'Loop iteration number {j}')
```

Similarly a `while` loop:
```python
i = 0
while i < 10:
    print(f'Loop iteration number {i}')
    i = i + 1
```
And finally, a function can be defined using the `def` keyword:
```python
# A function that adds two numbers and returns the result
def addition(a, b):
    # Add the numbers
    result = a + b
    # Return the result
    return result

# Run the function
addition(5, 10)
>> 15
```

Experiment with Python in the cell below. Learn to do the simple things that come to mind and take the chance to explore all of the options above.


```sos
%use sos

```

### <a class="anchor"></a> 2.A.a. NumPy

A very important addition to Python is the [NumPy](https://NumPy.org/) library, which provides multidimensional matrix/array processing tools. Most image processing libraries in Python use NumPy arrays to represent images. We will do the same, and therefore it is important you know how to use them.

One usually imports the NumPy library as `np`, to avoid typing `numpy` every time. This also makes the code more readable. Run the cell below to import NumPy.


```sos
%use sos
# Import the NumPy library
import numpy as np
```

To create a new NumPy array filled with zeros we can use `np.zeros`. It requires the size of the array we want to get as a tuple `(height, width)`. Run the cell below to see an example.

Optionally, we can specify the internal numerical [type](https://NumPy.org/devdocs/user/basics.types.html) of the array. For example, one can use 8-bit unsigned integers (`np.uint8`). This type is convenient for 8-bit images, as the resulting array will occupy less memory than with other types, such as `np.float32` or `np.float64`. For most of the labs, you will not need to mind the numerical type of your variables: it adapts automatically to represent what you are storing in each instance.


```sos
%use sos
# A NumPy array of size 5x10 with numerical type "unsigned 8-bit integer"
my_arr = np.zeros((5,10), dtype=np.uint8)
# Print the result
print(my_arr)
```

    [[0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]]


NumPy arrays can be accessed using the same notation used for lists, but with an index for every dimension. Before running the example below, try to guess where the `1` will appear in `my_arr`, i.e., what are $Y$ and $X$ in the comments.

<div class="alert alert-warning">
    
**Beware:** In NumPy arrays, the first index specifies the row (y-axis location), not the x-axis location, while the second index specifies the column (x-axis location), not the y-axis location.
</div>


```sos
%use sos
# Set the value at the Yth row, Xth column to 1
my_arr[3, 7] = 1
print(my_arr)
```

    [[0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 0 0 0]]


We can also use *slicing* (:) to determine whole regions of the array to read or assign. Before running the example, guess where the `1`s will appear, i.e., what are $Y_1,Y_2$ and $X_1,X_2$ in the comments.


```sos
%use sos
# Insert a rectangle with the upper left corner at (Y_1,X_1)
# and the lower right corner at (Y_2,X_2)
my_arr[:3, 2:6] = 1
print(my_arr)
```

    [[0 0 1 1 1 1 0 0 0 0]
     [0 0 1 1 1 1 0 0 0 0]
     [0 0 1 1 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 0 0 0]]


As you can see in the code, in slicing, we define an interval of numbers by `<lower_limit>:<upper_limit + 1>`. If one of the limits is not specified, it is interpreted as _from the beginning_ or _to the end_. 

Using slicing we can also extract patches of an array to create new arrays. Run the example below and see its effect.


```sos
%use sos
# Let's extract the rectangle that we created
my_rect = my_arr[:3, 2:6]
# Note: '\n' creates a new line when printing
print(f'Original array: \n{my_arr}\n\nExtracted patch: \n{my_rect}')
```

    Original array: 
    [[0 0 1 1 1 1 0 0 0 0]
     [0 0 1 1 1 1 0 0 0 0]
     [0 0 1 1 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 0 0 0]]
    
    Extracted patch: 
    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]


Using NumPy arrays we can easily perform the basic operators on all elements of one or several arrays (element-wise operations). Run the cell below to define a random array to work with.


```sos
%use sos
# First we define an array of size 5x5 filled with random integers from 0 to 10
np.random.seed(uid)
rand_array = np.random.randint(low=0, high=10, size=(5,5))
print(f'Initial array:\n{rand_array}')
```

    Initial array:
    [[6 5 4 2 1]
     [3 3 4 1 0]
     [7 5 5 9 8]
     [9 0 4 3 2]
     [0 8 0 8 4]]


We can perform any operation with a scalar for all elements of the array at the same time by using the array as if it was a variable with a single number. Run the example below and make sure you understand its results.


```sos
%use sos
# Add 2 to all elements
rand_array = rand_array + 2
print(f'rand_array + 2:\n{rand_array}\n')
# Divide all elements of the resulting array by 2
rand_array = rand_array / 2
print(f'rand_array / 2:\n{rand_array}')
```

    rand_array + 2:
    [[ 8  7  6  4  3]
     [ 5  5  6  3  2]
     [ 9  7  7 11 10]
     [11  2  6  5  4]
     [ 2 10  2 10  6]]
    
    rand_array / 2:
    [[4.  3.5 3.  2.  1.5]
     [2.5 2.5 3.  1.5 1. ]
     [4.5 3.5 3.5 5.5 5. ]
     [5.5 1.  3.  2.5 2. ]
     [1.  5.  1.  5.  3. ]]


Similarly, we can use expressions with multiple arrays of the same size and any of the basic operators: the calculation will be made on the co-positioned elements of the different arrays. Run the example below and make sure you understand its results.


```sos
%use sos
# Let's initialize two new random arrays
rand_array_1 = np.random.randint(low=0, high=10, size=(5,5))
rand_array_2 = np.random.randint(low=0, high=10, size=(5,5))
print(f'Array 1:\n{rand_array_1}\n\nArray 2:\n{rand_array_2}')
# We can for example add them together
added_arrays = rand_array_1 + rand_array_2
print(f'\nArray 1 + Array 2:\n{added_arrays}')
# Or multiply them
multiplied_arrays = rand_array_1 * rand_array_2
print(f'\nArray 1 * Array 2:\n{multiplied_arrays}')
```

    Array 1:
    [[3 1 1 2 7]
     [4 6 1 2 3]
     [8 3 0 7 0]
     [7 7 7 3 2]
     [9 9 5 9 5]]
    
    Array 2:
    [[4 3 5 0 8]
     [8 5 2 4 2]
     [4 1 3 0 8]
     [0 1 9 3 9]
     [6 5 4 4 1]]
    
    Array 1 + Array 2:
    [[ 7  4  6  2 15]
     [12 11  3  6  5]
     [12  4  3  7  8]
     [ 7  8 16  6 11]
     [15 14  9 13  6]]
    
    Array 1 * Array 2:
    [[12  3  5  0 56]
     [32 30  2  8  6]
     [32  3  0  0  0]
     [ 0  7 63  9 18]
     [54 45 20 36  5]]


We can also perform _advanced indexing_ of NumPy arrays: we can provide a [boolean](https://en.wikipedia.org/wiki/Boolean_data_type) array as the index and extract/modify only the values that are specified as `True`. An application of this form of indexing could be to take the multiplied array from the previous cell and set all its elements that are larger than $25$ to $100$. Run the example below and make sure you understand its results.


```sos
%use sos
# First print the multiplied array as a reference
print(f'Multiplied array:\n{multiplied_arrays}')
# Create the boolean array that contains true at every location, where the multiplied array has a value bigger than 25
boolean_array = multiplied_arrays > 25
print(f'\nBoolean array:\t(Multiplied array > 25)\n{boolean_array}')
# Now we can use this array to index the multiplied array and set all elements indicated by True to 100
multiplied_arrays[boolean_array] = 100
print(f'\nMultiplied array with all elements > 25 set to 100:\n{multiplied_arrays}')
```

    Multiplied array:
    [[12  3  5  0 56]
     [32 30  2  8  6]
     [32  3  0  0  0]
     [ 0  7 63  9 18]
     [54 45 20 36  5]]
    
    Boolean array:	(Multiplied array > 25)
    [[False False False False  True]
     [ True  True False False False]
     [ True False False False False]
     [False False  True False False]
     [ True  True False  True False]]
    
    Multiplied array with all elements > 25 set to 100:
    [[ 12   3   5   0 100]
     [100 100   2   8   6]
     [100   3   0   0   0]
     [  0   7 100   9  18]
     [100 100  20 100   5]]


Note that you can also create the boolean array in place, without explicitly declaring an additional variable. Run the cell below and see the difference with the preceeding cell. We will reuse the variable `multiplied_arrays`, and set all elements smaller than $20$ to $1$.


```sos
%use sos

# We get the array we have, and index only the elements smaller than 20.
multiplied_arrays[multiplied_arrays < 20] = 1
print(f'\nMultiplied array with all elements < 20 set to 1:\n{multiplied_arrays}')
```

    
    Multiplied array with all elements < 20 set to 1:
    [[  1   1   1   1 100]
     [100 100   1   1   1]
     [100   1   1   1   1]
     [  1   1 100   1   1]
     [100 100  20 100   1]]


As you can guess, these tools provided by NumPy are extremely useful to work on images. During the labs, you will get to know even more useful functions in the NumPy library, and they will make your life much easier.

### <a class="anchor"></a> 2.A.b. Matplotlib

In order to use the incredible tools NumPy provides on images, as well as to _see_ their effect, we need to be able to 1) load images as NumPy arrays and 2) visualize NumPy arrays as images. Here is where [Matplotlib](https://matplotlib.org/) comes into play, specifically the [PyPlot sublibrary](https://matplotlib.org/api/pyplot_api.html): it provides all the basic tools to load and visualize images.

One usually imports the library as `plt`, to avoid typing `matplotlib.pyplot` every time. This also makes the code more readable. Run the cell below to import Matplotlib.


```sos
%use sos
# Set the display to be inside the notebook
%matplotlib widget
# Import pyplot
import matplotlib.pyplot as plt
```

The only functionality you really need to know is how to load images using `plt.imread`:


```sos
%use sos
# Load an example image located at 'images/epfl.tif'
example_img = plt.imread('images/epfl.tif')
# Then, we can check the image is really a NumPy array
type(example_img)
```




    numpy.ndarray



Just so you can actually see the image you loaded, we will display it using `plt.imshow`. In the labs, you will never use this function. Instead, we provide the [IPLabViewer](#4_viewer) class, which adds extra functionality on top of `plt.imshow`, is more convenient, and is easier to use.


```sos
%use sos
# When using plt.imshow we first need to declare a new figure
plt.figure( )
# Then we add the image to the figure
plt.imshow(example_img, cmap='gray')
# Give it a title
plt.title(f"Example Image - SCIPER: {int(getpass.getuser().split('-')[2]) if len(getpass.getuser().split('-')) > 2 else ord(getpass.getuser()[0])}")
# And finally we make sure it is displayed on the screen
plt.show()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Pan', 'Pan axes with left…


Experiment with Python, NumPy and Matplotlib in the empty cell below. Learn to do the simple things that come to mind and take the chance to explore all of the options above. If you need some inspiration, run the next cell to follow the instructional video by Kay.


```sos
from IPython.display import IFrame
IFrame(width="800", height="450", src="https://www.youtube-nocookie.com/embed/oewtZoQb2gY")
```





<iframe
    width="800"
    height="450"
    src="https://www.youtube-nocookie.com/embed/oewtZoQb2gY"
    frameborder="0"
    allowfullscreen
></iframe>





```sos
%use sos

```

## <a class ="anchor"></a> 2.B. Basic JavaScript programming

Now let's look at the second programming language, JavaScript.
<div class="alert alert-info">
    
**Note:** If you are already familiar with JavaScript, you can skip to Section [2.C.](#-2.C.-Python-and-JavaScript-speed-comparison)
</div>

To declare a variable in JavaScript one uses the `var` statement. Similarly to what we saw with [Python](#-2.A.-Basic-Python-programming), we have three basic types of variables:
```javascript
// numbers
var my_var = 2;
var my_var = 5.2;
// strings
var my_var = 'Either use single quotes';
var my_var = "or use double quotes";
// boolean
var my_var = true;
var my_var = false;
```
Every variable can hold every type of data, so we don't need to specify the type at initialization.
<div class="alert alert-info">

**Note:** In JavaScript, every line should be terminated with a semicolon `;`.
</div>

We can have arrays of several items. There are two ways of constructing an array, the first one is:
```javascript
// list of numbers
var my_array = [2.5, 7, 3.3];
// list of strings
var my_array = ['E.T.', 'phone', 'home'];
// list of booleans
var my_array = [true, true, false];
```

And the second one is:
```javascript
// list of numbers
var my_array = new Array(2.5, 7, 3.3);
// list of strings
var my_array = new Array('E.T.', 'phone', 'home');
// list of booleans
var my_array = new Array(true, true, false);
```

Both are completely equivalent. 

<div class="alert alert-info">

**Note:** Note how with the second approach, you are explicitly calling the _constructor_ of the class Array (in other words, the function that creates new Arrays). Consequently, you need to use the keyword `new`. We will see this also in Section [3](#-3.-IPLabImageAccess) with the class we have prepared for you to process images simply with JavaScript.
</div>

Arrays are 0-indexed and can be accessed with square brackets:
```javascript
// set the 3rd element of my_list to 5
my_array[2] = 5;
// retrieve the 2nd element of my_list
var new_var = my_array[1];
```
Contrary to Python, tuples do not exist in JavaScript.

For numbers we have the following basic operators:
```javascript
// define the two variables
var a = 5;
var b = 2;
// addition
a + b;
>> 7
// subtraction
a - b;
>> 3
// multiplication
a * b;
>> 10
// division
a / b;
>> 2.5
// integer division can be emulated through
parseInt(a / b);
>> 2
// modulo (remainder of the integer division)
a % b;
>> 1
// power
a ** b;
>> 25
```
We can have conditional statements
```javascript
if(a < b){
    // printing in JS is done with console.log
    console.log('a is smaller than b');
}else{
    console.log('a is bigger than b');
}
```
<div class="alert alert-info">

**Note:** Indentation is **not** important in JavaScript but curly brackets `{}` are. To write a conditional statement, a loop, a function, etc., one uses curly brackets to define the beginning and end of each block of code. 
</div>

with the following comparison operators:
```javascript
<  // smaller than
<= // smaller or equal
>  // bigger than
>= // bigger or equal
== // equal
!= // not equal
```
A `for` loop is declared as:
```javascript
for(var i = 0; i < 10; i++){
    // another for loop inside the for loop
    for(var j = 0; j < 10; i++){
        // we can concatenate strings and numbers using the + operator
        console.log('Loop iteration number ' + j);
    }
}
```
Similarly a `while` loop:
```javascript
var i = 0;
while(i < 10){
    console.log('Loop iteration number ' + i);
    i++; // increment i by one
}
```

And finally, a function can be defined using the `function` keyword:
```javascript
// a function that adds two numbers and returns the result
function addition(a, b){
    // Add the numbers
    var result = a + b;
    // Return the result
    return result;
}

// run the function
addition(5, 10);
>> 15
```

<div class="alert alert-info">

**Note:** In the Jupyter Notebook environment we use for the labs, writing the `var` for variable declerations is not strictly necessary. Similarly, the semicolons at the end of every line are not needed for the code to work. That being said, it is still good practice to always add the two.
</div>

Experiment with JavaScript in the cells below. Learn to do the simple things that come to mind and take the chance to explore all of the options above. If you want to visualize what you try out, use the cell further below, Python, and SoS to play around. If that looks scary at this point, move on and come back some time later.


```sos
%use javascript
```


```sos
%use sos
```

## <a class="anchor"></a> 2.C. Python vs JavaScript speed comparison

As mentioned before, JavaScript will mainly be used to implement pixel-level algorithms, which require nested `for` loops. To illustrate why this is necessary, we'll compare the time it takes for both Python and JavaScript to perform a zero-padded convolution implemented using nested `for` loops.

The two-dimensional convolution between two images `h` and `f` is given by
$$(h \ast f)[k,l] = \sum_{m \in \mathbb{Z}}\sum_{n \in \mathbb{Z}}f[m,n]h[k-m,l-n]$$
Don't worry, you don't need to know how that works right now, you'll see this in Chapter _3.2_ of the course. But it's safe to say that the convolution is a very important operation for image processing, which is why we'll use it for this demonstration.

First let's define a Python function to generate a Gaussian kernel, which we will use as one of the images for the convolution. Run the cell below to define this function and obtain and visualize the Gaussian impulse response.


```sos
%use sos
# Imports and definitions, in case you skipped the previous parts
%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
import getpass

# Function that creates a Gaussian convolutional kernel based on the input parameters
def gaussian_kernel( sig, length):
    # Generate the 1D grid-points
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    # Generate the 2D grid-points created by ax x ax
    xx, yy = np.meshgrid(ax, ax)
    # Calculate the Gaussian kernel (up to proportionality constant)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    # Normalize the kernel to sum 1 and return it
    return kernel / np.sum(kernel)

# Get a Gaussian kernel
ker = gaussian_kernel(sig=3, length=5)
# Show the result
plt.figure()
plt.imshow(ker, cmap='gray')
plt.title(f"Gaussian kernel - SCIPER: {int(getpass.getuser().split('-')[2]) if len(getpass.getuser().split('-')) > 2 else ord(getpass.getuser()[0])}")
plt.show()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Pan', 'Pan axes with left…


From the implementation above you can learn how to use NumPy functions such as `np.linspace`, `np.meshgrid`, `np.exp`, `np.square` and `np.sum`. Try them out at the end of the Python section if you are curious! Otherwise, don't worry, this example will always be here for you.

Now that we have the Gaussian kernel ready, let's define the JavaScript convolution function. Run the cell below to define it.


```sos
%use javascript
// function that convolves img with kernel
function convolution(img, kernel){
    // initialize the output image as a copy of the input image
    var smoothed_img = [...img];
    // compute the kernel offset
    const k_offset = Math.floor(    kernel.length / 2 );
    const l_offset = Math.floor( kernel[0].length / 2 );
    // perform zero-padded convolution
    for(i = 0; i < img.length; i++){
        for(j = 0; j < img[0].length; j++){
            // compute output for pixel i, j. 
            // zero-padding: extend the image with zeros beyond its boundaries.
            smoothed_img[i][j] = 0;
            for(k = 0; k < kernel.length; k++){
                for(l = 0; l < kernel[0].length; l++){
                    // compute evaluation points to enforce boundary
                    var x = k - k_offset + i;
                    var y = l - l_offset + j;
                    // increment if appropriate (if outside the image, don't increment, there are zeros)
                    if((x < img.length) && (x > 0) && (y < img[0].length) && (y > 0)){
                        smoothed_img[i][j] += img[x][y] * kernel[k][l]; 
                    } 
                }
            }
        }
    }
    // return the output image
    return smoothed_img;
}
```

From the implementation above, you can learn useful JavaScript functions like `Math.floor`. Nonetheless, some of the tricks we had to use will not be necessary for you. For example, the lines
```javascript
var smoothed_img = [...img];
k < kernel.length;
l < kernel[0].length;
```
will be much more intuitive to implement using our class [IPLabImageAccess](#-3.-IPLabImageAccess-(2-points)) below, so do not worry about them. 

Now, let's define the Python convolution function. Run the cell below to define it.


```sos
%use sos
# Function that convolves img with kernel
def convolution(img, kernel):
    # Initialize output image as an empty image
    smoothed_img = np.zeros(img.shape)
    # Compute the kernel offset
    k_offset = kernel.shape[0] // 2
    l_offset = kernel.shape[1] // 2
    # Perform zero-padded convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Compute output for pixel i, j. 
            # Zero-padding: extend the image with zeros beyond its boundaries.
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    # Compute evaluation points to enforce boundary
                    x = k - k_offset + i
                    y = l - l_offset + j
                    # Increment if appropriate (if outside the image, don't increment, there are zeros)
                    if (x < img.shape[0]) and (x > 0) and (y < img.shape[1]) and (y > 0):
                        smoothed_img[i, j] += img[x, y] * kernel[k, l]
    # Return the output image
    return smoothed_img
```

From the implementation above, you can learn useful NumPy functions like `np.zeros`, or properties like `img.shape`, when `img` is a NumPy array. As always, feel free to try them out now at the end of the Python section!

Now, we load an image and transfer it to JavaScript, together with the Gaussian kernel we defined before in Python. In this way, we will be able to test the JavaScript function we implemented. Run the cell below to load the image, display it, and transfer the image and the Gaussian kernel to Javascript.


```sos
%use sos
%put hrct --to javascript
%put ker --to javascript
# Load the hrct image
hrct = plt.imread('images/hrct.tif')
# Display it
plt.figure()
plt.imshow(hrct, cmap='gray')
plt.title(f"HRCT - SCIPER: {int(getpass.getuser().split('-')[2]) if len(getpass.getuser().split('-')) > 2 else ord(getpass.getuser()[0])}")
plt.show()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Pan', 'Pan axes with left…


Now, we are ready to run the convolution in JavaScript and measure the time it takes. Run the cell below to do it.


```sos
%use javascript
%put convolved_hrct_js 
%put js_time 
// get the starting time in miliseconds
var start = new Date();

// run the convolution
var convolved_hrct_js = convolution(hrct, ker);

// get the ending time in miliseconds
var end = new Date();

// calculate the elapsed time in miliseconds
var js_time = end - start

// display the elapsed time
console.log('The JavaScript convolution took ' + js_time + ' miliseconds to run.');
```

    The JavaScript convolution took 313 miliseconds to run.


Now, we do the same for the implementation of the convolution in Python. Run the cell below to do it.
<div class="alert alert-danger">

**Spoiler alert:** You can go grab a drink while this is running... As you may have noticed, in a Jupyter Notebook a cell is still running if inside the square brackets in the top left corner of the cell you see an asterisk `[*]` instead of a number.
</div>

<div class="alert alert-info">

**Note:** To measure the time in Python we need to import the `time` module.
</div>


```sos
%use sos
import time

# Get starting time in seconds
start = time.time()

# Run the convolution
convolved_hrct_jpython = convolution(hrct, ker)

# Get the ending time seconds
end = time.time()

# Calculate the elapsed time in miliseconds
python_time = round((end - start)*1000)

# Display the elapsed time
print(f'The Python convolution took {python_time} miliseconds to run.');
print(f'That makes JavaScript around {round(python_time/js_time)} times faster than Python for this implementation.')
```

    The Python convolution took 45206 miliseconds to run.
    That makes JavaScript around 144 times faster than Python for this implementation.


Hopefully, you can understand from this illustration that standard Python is not made for pixel-level algorithms with nested `for` loops. This justifies the use of JavaScript for such implementations. Just so you can actually see the result of the function we just ran, run the next cell to see it.


```sos
%use sos
plt.figure()
plt.imshow(convolved_hrct_jpython, cmap='gray')
plt.title(f"HRCT convolved with a Gaussian kernel - SCIPER: {int(getpass.getuser().split('-')[2]) if len(getpass.getuser().split('-')) > 2 else ord(getpass.getuser()[0])}")
plt.show()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Pan', 'Pan axes with left…


# <a class="anchor"></a> 3. IPLabImageAccess (2 points)

While JavaScript offers the big advantage of fast nested `for` loops for pixel-by-pixel programming, it also comes with a downside. It does not offer convenient array/image processing libraries like Python. To make things easier, we created our own JavaScript class, `IPLabImageAccess`, which facilitates the handling of images in JavaScript. Feel free to check out the documentation [here](https://github.com/Biomedical-Imaging-Group/IPLabImageAccess/wiki). Every `IPLabImageAccess` object is a container for an image and provides easy-to-use methods for pixel access that take care of boundary conditions. This will help you to complete the tasks in JavaScript. Here, we will show you the basics of how this class can be used.

As we do with external libraries, we will import this class under a simpler name to make the code more readable. In all upcoming labs, we will import the `IPLabImageAccess` class as `Image`. Run the JavaScript cell below to do it.


```sos
%use javascript
var Image = require('./lib/IPLabImageAccess.js')
```

Once imported, we can start using it. First of all, let's look at the different ways we can create a new `Image` object.

* An `Image` object can be created with
```javascript
var img = new Image(height, width);
```
This will create a new `Image` object of dimensions `height` $\times$ `width`. <br><br>

* Optionally, `height` and `width` can be provided as a single array `[height, width]`. This makes it easy to create an image with the same dimensions as an existing image by using the `shape` method, for example
```javascript
var img = new Image(an_existing_image.shape());
```

* By default, the image is initialized to $0$ for every pixel, but we can choose an alternative initialization value. For this, we need to create an `options` object in which we specify the `init_value` attribute and then we pass this object to the `Image` constructor
```javascript
var options = {};
options.init_value = 255;
var img = new Image(height, width, options);
```

* Similarly, if we want to create an rgb image, we specify the `rgb` attribute in the `options` object to `true` and pass it to the constructor.
```javascript
var options = {};
options.rgb = true;
var img = new Image(height, width, options);
```

* An `Image` object can also be created from an existing JavaScript array, by simply passing it as a single argument
```javascript
var img = new Image(an_existing_array);
```
This is useful, for example, to create an `Image` object from a Python NumPy array converted to Javascript.

We now look at how to use the basic functionalities in the class. The most important operations you're gonna need are the `img.setPixel` and `img.getPixel` methods, which do exactly as their name suggests when `img` is an `Image` object.
* The `img.setPixel(x, y, value)` method can be used to set the pixel at location `(x,y)` to a specific value `value`:
```javascript
// here we set the pixel (3,5) of img to 255
img.setPixel(3, 5, 255);
```
* The `img.getPixel(x, y)` method can be used to retrieve the value of the pixel at location `(x,y)`:
```javascript
// here we retreive the pixel (3,5) of img
var pixel = img.getPixel(3, 5);
```
By default, the `setPixel` and `getPixel` methods will use _mirroring_ boundary conditions to provide values for pixels outside the image range. Do not worry if you do not understand this yet, it will be clearer as you progress through the course.<br><br>

* The height and width of an `Image` object can be retreived either using using the `img.shape()` method, which returns a list `[height, width]` or, more conveniently, with `img.nx` and `img.ny`.
```javascript
// retreive height and width of the image as a list
height_and_width_list = img.shape();
// get only the width
width = img.nx;
// get only the height
height = img.ny;
```
This second option is very useful when setting the limits in `for` loops. <br><br>

* We can extract the neighbourhood of size `width` $\times$ `height` around the pixel at location `(x,y)`, using the `img.getNbh(x, y, width, height)` method:
```javascript
// extracting the 3x3 neighbourhood around pixel (12,7)
nbh = img.getNbh(12, 7, 3, 3);
```
By default, the method will use _mirroring_ boundary conditions, just like the `getPixel` and `setPixel` methods.

* Finally, if we need to convert an `Image` object to a simple JavaScript array, we can do it with the `img.toArray()` method, for example
```javascript
// convert the Image object to a JavaScript array
img_array = img.toArray();
```
This will be necessary, for example, to convert the image to Python.

We are sorry you just had to read so much! For **2 points**, your job now is to create a new `Image` object called `img`, of size $50 \times 50$, and set the pixels at locations $(x=10, y=20)$, $(x=15, y=30)$ and $(x=40, y=7)$ to $255$.


```sos
%use javascript
%put img 

// create a new Image object named img of size 50 x 50
var img = new Image(50, 50);

// set the pixel at location (x=10,y=20) to 255
img.setPixel(10, 20, 255);
// set the pixel at location (x=15,y=30) to 255
img.setPixel(15, 30, 255);
// set the pixel at location (x=40,y=7) to 255
img.setPixel(40, 7, 255);
```

Run the cell below to (partially) check if you did the right thing.


```sos
%use javascript
if( (img.nx != 50) || (img.ny != 50) ){
    throw new Error("The image size is not correct.");
}else{console.log('Good job! Your image has the correct size.')}

if( img.getPixel(40,7) != 255 ){
    throw new Error("The pixel at location (x=40, y=7) should have value 255.")
}else{console.log('Good job! The pixel at (x=40, y=7) was properly set.')}

```

    Good job! Your image has the correct size.
    Good job! The pixel at (x=40, y=7) was properly set.


Experiment with the `IPLabImageAccess` class in the empty cell below. Learn to do the simple things that come to mind and take the chance to explore all of the options above. If you need some inspiration, run the next cell to follow the instructional video by Kay.


```sos
from IPython.display import IFrame
IFrame(width="800", height="450", src="https://www.youtube-nocookie.com/embed/x1hsW-fWAi8")
```





<iframe
    width="800"
    height="450"
    src="https://www.youtube-nocookie.com/embed/x1hsW-fWAi8"
    frameborder="0"
    allowfullscreen
></iframe>





```sos
%use javascript

```

# <a class="anchor"></a> 4. IPLabViewer

As you saw in Section [2.A.a](#-2.A.a.-NumPy), the main reason that Python is so powerful is because of its libraries. To ease image visualization and manipulation, we have developed a library, the `IPLabViewer` class, which you will be using throughout the labs. This section explains its functioning and shows some examples and use-cases of the `IPLabViewer`. Before using the class, be sure to check the documentation. You can call `help(viewer)` in a cell, but we recommend you to check its extensive [wiki](https://github.com/Biomedical-Imaging-Group/IPLabImageViewer/wiki/Python-IPLabViewer()-Class). 

The IPLabViewer runs without explicitly importing any external libraries (all are imported in the class' `.py` file). However, we will import the library Ipywidgets ([see documentation](https://ipywidgets.readthedocs.io/en/latest/)) to allow for interactive visualization. You will even use it to add your own functions to the viewer's interface! 

<div class = "alert alert-info">

**Importing `IPLabViewer`**: To start using you need to activate Matplotlib's dynamic environment, with the magic command `%matplotlib widget`, and import the library IPLabViewer. We will always import it as `viewer`, using the following two lines:

```python
%matplotlib widget
from lib.iplabs import IPLabViewer as viewer
```
</div>
    
Run the cell below to import the viewer and load the images we will work with.


```sos
%use sos
# Configure plotting as dynamic
%matplotlib widget
# Import required packages for this exercise
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
# Import IPLabViewer
from lib.iplabs import IPLabViewer as viewer

# Import images
car   = plt.imread('images/car_pad.tif')
hrct  = plt.imread('images/hrct.tif')
hands = plt.imread('images/hands.tif')
plate = plt.imread('images/plate.tif')
boats = plt.imread('images/boats.tif')
```

## <a class="anchor"></a> 4.A. Creation of a viewer

The cell after the next one will illustrate many of the functionalities of IPLabViewer, as well as its customization capabilities. Run it and explore its results while reading the explanatory cell below it. Feel free to run the cell immediately below this one before to follow the instructional video by Alejandro Noguerón Arámburu, one of the student developers behind these labs. He will guide you and exemplify the use of the viewer.


```sos
from IPython.display import IFrame
IFrame(width="800", height="450", src="https://www.youtube-nocookie.com/embed/OKYy3C0MCm0")
```





<iframe
    width="800"
    height="450"
    src="https://www.youtube-nocookie.com/embed/OKYy3C0MCm0"
    frameborder="0"
    allowfullscreen
></iframe>





```sos
%use sos

# Declare image list
first_list = [hands, plate, car, hrct]
# Declare titles (only two)
title_list = ['Hands', 'Plate']
# First we close all matplotlib existing figures to improve performance
plt.close('all')
# Call the image viewer with the image list as first argument. The rest of the arguments are optional
first_viewer = viewer(first_list, title = title_list, colorbar = True, widgets = True, 
                      hist = True, axis = True, cmap = 'spring')
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))


To create an `IPLabViewer` object, you simply call the function `viewer` and assign it a variable name. Ideally, the variable should be self-explanatory and end with the word *viewer* to avoid confusion (for example, `first_viewer` as above).

* Images: The same viewer can load many images at once. In this example, it was created with $4$ images. By default, you can switch between the images using the buttons `Prev` and `Next` on the widgets panel.
* Titles: You can give the viewer titles for each image, which will be shown immediately above it. If you do not provide a title for some images, the variable name will be shown by default. In this example, only $2$ were given.
* Customization: Several of the class' functionalities (e.g., `widgets`, `hist`, `axis`, `colorbar`) can be enabled from the beginning. A user can later turn them off using the widgets or [calling methods](#prog_customization) of the object. 
* Colormap: The colormap to be used can be set from the start with the parameter `cmap = 'colormap'`, and dynamically changed using `Options` and choosing one of the supported colormaps in the drop-down menu. 
* Statistics: On the lower part of the widget panel, the image statistics of the area of the image being displayed are shown. If you use the button with a little square at the left of the images to zoom into a selected region, you will see the statistics are updated in real time. You can also use the button with crossed double-arrows to pan through the image at a specific level of zoom. You can always use the button *Reset* to see the whole image again.

<div class="alert alert-warning">
    
<b>Beware: </b> The statistics displayed in the `IPLabViewer` objects are rounded to the second decimal. If you need an exact statistic, use the methods `np.mean()` or `np.std()` on the images (NumPy arrays) directly.
</div>

* Histogram: Besides the normal *counts* vs *value* bars in a histogram, you will see a black line. This line represents *input* vs *output* intensity values. This line changes its position and slope if you change the brightness and contrast of the image. All pixel values that are to the left of the point where the line touches the bottom have the minimum intensity, and all that are to the right of the point where the line touches the top have the maximum intensity, and the ones that are in-between those two points have an intensity that is linearly scaled between the maximum and minumum value.

* Display mode: There are several options to display a list of images
    * Default display: Display only one image at a time, and use the buttons *Next* and *Prev* to browse the different images. 
    * Set `subplots = [m, n]`: Arrange the images in an $m \times n$ grid.

## <a class="anchor"></a> 4.B. Using widgets

If you just want a quick visualization, you can call the `IPLabViewer` with the image you want to display (or list of images). All the parameters will be set to their default values (`cmap = 'gray'`, single image mode, all the features set to off), but you can use the widgets to change this. Run the next cell to see all the default values. Go through all the widgets and explore their options while going through the explanatory cell below.


```sos
%use sos

# In order to keep the memory load on Noto low,
# we will generally close all figures before continuing.
# This means that the previous viewers are no longer interactive!
# But you can run their cell again if you need to.
plt.close('all')

# Declare image list
second_list = [hrct, hands, car]
# Call the image viewer with minimal arguments
second_viewer = viewer(second_list)
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


Your first view will consist of:
* A **toolbar** with series of buttons at the left, which serve to control the dynamic environment. If you hover your mouse in the buttons, wou will see what each button does:
    * *Three horizontal lines*: Hide the toolbar
    * *House*: Reset the zoom of the images. 
    * *Cross with arrows*: Pan through an image. Click on it, then click on the image, hold and drag to pan through the axis.
    * *Square*: Zoom. Click on it, then click on the image, hold and drag to create a rectangular area. Release to zoom to the region.
    * *Floppy disk*: Take screenshot of figure and save in PNG format. 
* The **figure** holding the images (one figure is holding all the images). You can change the size of a figure using the small gray triangle (bottom-right corner). If you hover your cursor over the triangle, you will see it change to a two-sided arrow. By clicking on it you will be able to adjust the figure size.
* A **button** with the legend *Show Widgets*. Click on it and it will take you to the widget main menu. 

In the widgets main menu you will see the following widgets:
* **Brightness and Contrast** Slider: In this menu you will be able to change the color scaling of the image through a slider, given in percentages of the original maximum and minimum intensities.
* **Show Histogram** Button: Show or hide the histograms of the images.
* **Options** Button: It will take you to the options menu, where you will be able to show or hide the axis, the colorbar, and change the colormap.
* **Reset** Button: This will reset the all the parameters to the default state (colormap, brightness/contrast ect.), while the _House Button_ on the top left only resets the display area of the viewer and doesn't change any parameters.
* **Stats** Textbox: Here you will see the mean, the standard deviation, the range of values and the size of the image. Mean, standard deviation and range of values will be updated when you zoom to a region. 

### Multiple choice question

Now, for **1 point**, and to finish the section on the `IPLabViewer` class, answer the following question using the viewer that you just defined.

In the image `car`, what is the mean value of **the carpet that the car is standing on**? Hint: Zoom into several regions that are not too close to the edge nor to the car.

1. Around 40
2. Around 60 
3. Around 80

In the following cell, modify the variable `answer` to your actual answer, e.g., `answer=1`, `answer=2`, or `answer=3`.


```sos
%use sos
# Assign your answer to this variable
answer = 3
```


```sos
%use sos
# Let's do a sanity check - check that the answer is one of the possible choices
assert answer in [1, 2, 3], "Select one of the answers: 1, 2 or 3."

```

## <a class="anchor"></a> 4.C. User defined widgets

`IPLabViewer` has several cool functionalities. But what if you wanna contribute yourself? `IPLabViewer` allows you to create a function in a Jupyter notebook, and apply it simultaneously to all images within your `IPLabViewer` object for a set of slider values. The function you create will take as parameters:
* an image (`NumPy array`), and 
* one or more numerical values.

Your function is then supposed to apply an operation on the image that depends on these value(s). This is very useful if you want to see the effect of the value(s) you choose on one or several images. Usually, you would have to manually run the same process several times, and visualize the results each time. With `IPLabViewer`, you can simply use slider(s) within the viewer.

In the next cells, you will see a very basic example. We will apply a pixel-wise operation on the image `car`: All the pixels with a value below a treshold (given in $\%$) will be set to the *maximum* value. We will guide you through the process.

First, we define the thresholding function `your_function`. Run the next cell to define it.


```sos
%use sos
# Define your function
def your_function(image, threshold):    
    # We make a copy of the original, where we will apply our threshold.
    output = np.copy(image)
    # For greater flexibility, we will get the value as a percentage of the maximum
    value = threshold*0.01*np.amax(image)
    # Apply threshold
    output[image < value] = np.amax(image)
    return output
```

Besides defining your function, you need to define the corresponding slider(s), a button to run `your_function` with the sliders' values, and the specific function that will run `your_function` (a.k.a., the slider's _callback_ function).

<div class = "alert alert-info">
    
*Note*: Although this is rather abstract, the example will make it clear. We will use an integer slider in the range $[0,100]$ (`widgets.IntSlider`), a button that clearly specifies what our function does (`widgets.Button`), and a callback function `callback` that calls `your_function` with the slider's value as threshold. Run the next cell, click the button *Extra Widgets*, and try different values. Remember to click on `Apply Threshold`!
</div>
    
If you want some guidance, run the next cell to follow the instructional video by Alejandro.


```sos
from IPython.display import IFrame
IFrame(width="800", height="450", src="https://www.youtube-nocookie.com/embed/__JXopJPLxQ")
```





<iframe
    width="800"
    height="450"
    src="https://www.youtube-nocookie.com/embed/__JXopJPLxQ"
    frameborder="0"
    allowfullscreen
></iframe>





```sos
%use sos
# close all open figures
plt.close('all')

# Declare slider
threshold_slider = widgets.IntSlider(value = 0,min = 0, max = 100, step = 1, 
                                     description = 'Threshold', continuous_update = False )

# Declare button with meaningful description
activation_button = widgets.Button(description = 'Apply Threshold')

# Declare callback
def callback(img):
    # Get slider value
    threshold = threshold_slider.value
    # Call your function
    output = your_function(img, threshold)
    return output

# Call viewer, passing the widget and callback separately as lists
threshold_viewer = viewer(car,  new_widgets = [threshold_slider, activation_button], callbacks = [callback], widgets = True)
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))


If you're interested in this and similar topics, we invite you to check the documentation for Ipywidgets [here](https://ipywidgets.readthedocs.io/en/stable/).

## <a class="anchor"></a> 4.D. Programmatic customization

Most options can be modified both through widgets and _programmatically_ (in code). Run the following cells. In the first one, we will create the object `car_viewer`, which displays the image `car`. By running the next cells, one by one, we will produce changes on the `car_viewer` without interacting with it directly. 


```sos
%use sos
plt.close('all')
car_viewer = viewer(car)
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())



```sos
%use sos

car_viewer.set_axis(axis = True)
```


```sos
%use sos

car_viewer.set_widgets(widgets = True)
```


```sos
%use sos

car_viewer.set_colormap(colormap = 'viridis')
```


```sos
%use sos

car_viewer.show_histogram(hist = True)
```


```sos
%use sos

car_viewer.set_axis(axis = False)
```


```sos
%use sos

car_viewer.set_colorbar(colorbar = True)
```

## <a class="anchor"></a> 4.E. Try it yourself!

### Creating an IPLabViewer (1 point)
Now that you have explored the capabilities of the `IPLabViewer` class, for **1 point**, create in the next cell a viewer object with the following characteristics (do it with code, on one or several lines):
* The first image should be `hrct` and the second `plate`
* The titles are left as default
* The colormap is set to 'inferno'
* There is a colorbar 
* The axis are set to `False`
* Only a single image at a time is displayed

Store your object in the variable `exercise_viewer`.


```sos
%use sos
plt.close('all')
exercise_viewer = viewer([hrct, plate], subplots=[1,2])
exercise_viewer.set_colormap(colormap = 'inferno')
exercise_viewer.set_colorbar(colorbar = True)
car_viewer.set_axis(axis = False)
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


If you want to check the state of your `IPLabViewer` object, you can check [the documentation](https://github.com/Biomedical-Imaging-Group/IPLabImageViewer/wiki/Python-IPLabViewer()-Class) and explore the attributes. See the two examples in the next cell. Both should show that there are indeed two images. The second one is in the form of an assertion (it will produce an error if the condition is not met), to let you know if your viewer is in the right direction.


```sos
%use sos

print(len(exercise_viewer.image_list))
assert exercise_viewer.number_images == 2, f'Your viewer should have 2 images! Instead it has {exercise_viewer.number_images}.'
print('Well done! Your viewer seems to be working.')
```

    2
    Well done! Your viewer seems to be working.


### Threshold operation (2 points)

Now you will program a function and add a slider to apply this function to the image (see Section [4.C.](#-4.C.-User-defined-widgets)).

For **1 point**, your first task is to define a function `range_function(img, rmin, rmax)` that will only **keep** pixel values inside a range of values $[r_{\mathrm{min}}, r_{\mathrm{max}}]$ ($r_{\mathrm{min}}$ and $r_{\mathrm{max}}$ included), while the values outside this range will be set to $0$.
You will apply it on the image `car`, so `rmin` and `rmax` should be between $0$ and $255$. Define this function in the following cell. 

<div class="alert alert-info">
  
<b>Note: </b> Use boolean indexing of the NumPy array `img`, as in the thresholding example, to perform the necessary operations.
</div>


```sos
%use sos

# Define your function
def range_function(img, rmin, rmax):    
    # Best practices are creating a copy. You don't wanna mess the original.
    output = np.copy(img)
    # set the pixels of output that are smaller than rmin or bigger than rmax to 0
    for i in range(output.shape[0]):
            if (output[i] < rmin or output[i] > rmax) :
                output[i]=0
    return output
```

You can test your function with the following cell. In it, we define a linear array `test_array` that is formed by the integers from 0 to 255 in ascending order, using the method [NumPy.arange](https://NumPy.org/doc/stable/reference/generated/NumPy.arange.html). Then, we apply your function with the range $[100, 199]$ to it. There should be exactly $100$ elements that were not set to $0$. If the cell runs smoothly, your implementation is probably correct.
<div class="alert alert-info">
    
**Note:** The function `np.count_nonzero(arr)` counts the number of elements in `arr` that are not zero.
</div>


```sos
%use sos

test_array = np.arange(0, 256)
assert np.count_nonzero(range_function(test_array, 100, 199)) == 100, 'Your implementation is not yet correct.'
print('Congratulations! Your function is correct.')

```

    Congratulations! Your function is correct.


For **1 point**, declare the necessary widgets to control this function (slider(s) and a button, as well as the button's callback function). Use one `IntRangeSlider` ([see doc](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#IntRangeSlider)). When you get the value of this slider (by calling `IntRangeSlider.value`), you get a list with two elements corresponding to the minimum and the maximum value of the range. You can use these values to call your function (`range_slider.value[0]` and `range_slider.value[1]` if `range_slider` has been defined as an `IntRangeSlider` widget). 

Use the next cell to declare your widgets (store the slider in the variable `range_slider`, your button on `range_button` and your callback on `range_callback(img)`)


```sos
%use sos

# Declare slider
range_slider = widgets.IntSlider(value = 0,min = 0, max = 100, step = 1, 
                                     description = 'Threshold', continuous_update = False )

# Declare button with meaningful description
range_button = widgets.Button(description = 'Apply Threshold')

# Declare callback
def range_callback(img):
    # Get slider value
    threshold = range_slider.value
    # Call your function
    output = range_function(img, threshold)
    return output
```

Now we will create an `IPLabViewer` using your widgets. Explore the results of your function (click the button *Extra Widgets*), explore different values and see if you implemented it correctly.


```sos
%use sos

# Call viewer, passing the widget and callback separately as lists, e.g., range_viewer = viewer(car, ...)
plt.close('all')
range_viewer = viewer(car,  new_widgets = [range_slider, range_button], callbacks = [range_callback], widgets = True)

```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))


# <a class="anchor"></a> 5. Image processing in Python (2 points)

A goal of the labs is for you to learn the usual ways to perform image-processing tasks in Python. Therefore, after you fully understand the nature of every algorithm, and have implemented them yourself in JavaScript, the labs will always move towards trustworthy implementations in popular image processing libraries. There are many libraries available, but throughout this course we will focus on three of the most common:

* [**`openCV`**](https://opencv.org/about/): It stands for Open Computer Vision. As the name suggests, it provides thousands of open source algorithms for computer vision applications. The fact that it supports several programming laguages (`C++`, `Python`, `Java` and `Matlab`), and that it has over 20 years of history explains the huge community of users. Thus, there is also extensive documentation and tutorials, not only in its official website, but also on several other sites.  
* [**`scipy.ndimage`**](https://www.scipy.org/) SciPy stands stands for Scientific Python, and it is one of the most widely used Python libraries (their packages include `NumPy` and `Matplotlib`, which are considered standards in Python). As OpenCV, it is open-source and it has a huge community, as well as extensive documentation and tutorials. It has a myriad of applications for scientific and technical computing. In the IP courses we will focus on their module built for [Multi-dimensional image processing (`ndimage`)](https://docs.scipy.org/doc/scipy/reference/ndimage.html).
* [**`scikit-image`**](https://scikit-image.org/) Scikit stands for SciPy Toolkits. These toolkits are add-on packages for SciPy. They are developed and maintained individually. As SciPy, they are open-source projects. Some of these Scikits have become very well known, and have a large community of users and developers. Scikit-Image is one of them, developed specifically for image-processing tasks.

While there are other good libraries for image processing, we will focus on these three because their history and community ensure long-term support and extensive documentation and examples. Whenever we want you to use one of these libraries, we will either use a function to exemplify an operation, or we will include a link to the documentation of that particular function.

For a short instructional video by Alejandro on how to check the documentation in each of these three libraries run the cell below.


```sos
from IPython.display import IFrame
IFrame(width="800", height="450", src="https://www.youtube-nocookie.com/embed/MmBIn6vcCqM")
```





<iframe
    width="800"
    height="450"
    src="https://www.youtube-nocookie.com/embed/MmBIn6vcCqM"
    frameborder="0"
    allowfullscreen
></iframe>




Just as with `NumPy` (`np`) or `matplotlib.pyplot` (`plt`), there are standard abbreviations to import these libraries. Run the next cell, where we will import the three of them. As you will see, the packages do not necessarily keep the name of the organization. 


```sos
%use sos
# OpenCV
import cv2 as cv
# SciPy NDImage
import scipy.ndimage as ndi
# Scikit-image
import skimage
```

# <a class="anchor"></a> 5.A. Examples

Now we will go through two very basic operations (rotation and edge detection) for you to get familiar with each of these libraries. We will show you the basic syntax, or how to call the function with the minimum necessary parameters. For the full list of parameters and options, go to the official documentation of each function. 

<div class = "alert alert-warning">
  
**Note**: Did you notice how in the `viewer` we specified all the parameters but the first one? In image processing libraries, the first parameter is almost always the input image. However, the rest of the parameters may vary from function to function and from library to library. It is best practice to specify them, otherwise, the order of the parameters becomes important and you might get confused with the way you call the function.   
</div>

### Rotation in OpenCV

The basic syntax of the function is (see [the documentation](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga4ad01c0978b0ce64baa246811deeac24)):
```python
output = cv.rotate(src, rotateCode)
```

The parameters are:
* `src` (`numpy` array): The source image, and
* `rotateCode` (numeric flag): Specifies the rotation (see the available options [here](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga6f45d55c0b1cc9d97f5353a7c8a7aac2)).

As you will see, OpenCV often works with these kind of flags, so you will have to check the documentation of each particular function to know their possible values. Run the next cell to see how this function works.


```sos
%use sos

# First, we call the OpenCV function
rot_car = cv.rotate(car, rotateCode =  cv.ROTATE_90_CLOCKWISE)

# and then we visualize
plt.close('all')
cv_rotate_viewer = viewer([car, rot_car], title = ['Original', 'Rotated'], subplots=[1,2])
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


### Edge detection in OpenCV

For a quick edge detection filter, we will use the Sobel operator *derivative* filter. You will go through it in detail during the course.

The syntax of the function with the basic parameters is (see [the documentation](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)):
```python
output = cv.Sobel(src, ddepth, dx, dy, ksize, borderType)
```

The parameters are:
* `src` (`numpy` array): source image
* `ddepth` (flag): The data type of the pixels. As a rule of thumb, you can use `cv.CV_32F`
* `dx` (integer): Order of derivative in x
* `dy` (integer): Order of derivative in y
* `ksize` (integer): The size of the kernel. Defaults to 3 (standard Sobel filter), but OpenCV gives you the option to use extended kernels (1, 3, 5 or 7)
* `borderType` (numerical flag): Specifies the boundary conditions (see the available options [here](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=copymakeborder)). Throughout the course, we will be using `cv.BORDER_REFLECT`: mirroring boundary conditions are applied when values outside the image are needed.

Run the next cell for an example on the image `plate`.


```sos
%use sos

# We apply the function on the image
edges_plate = cv.Sobel(plate, ddepth = cv.CV_32F, dx =  1, dy = 1, ksize = 3, borderType = cv.BORDER_REFLECT)

# And compare to the original
plt.close('all')
cv_sobel_viewer = viewer([plate, edges_plate], title = ['Original', 'Edges'], subplots=[1,2])
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


### Rotation in SciPy NDImage

The basic syntax of the function for grayscale images is  (see [the documentation](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.rotate.html)):
```python
output = scipy.ndimage.interpolation.rotate(input, angle, reshape=True, mode='reflect')
```

The parameters are:
* `input` (`numpy` array): The source image, 
* `angle` (floating point number): Rotation angle in degrees,
* `reshape` (boolean): If `True` (default), the output shape is adapted so that the input array is contained completely in the output.
* `mode` (string): Specify boundary conditions. We will use `'reflect'` (default value) throughout the course: mirroring boundary conditions are applied when values outside the image are needed. Other options include `'constant'`, `'nearest'`, or `'wrap'`.

You can already see some differences between the two libraries: `scipy.ndimage` does not work with flags, but with strings. You will still have to check the documentation to see the available options. Moreover, in this particular function, SciPy allows rotation by any angle. 

Run the next cell to see the result of rotating the image `car` for an angle $\theta = 45^\circ$. In the stats box of the `IPLabViewer`, look at the size of both images. As you can see, the resulting image has been adapted. Modify the cell, and select `mode = 'constant'`. See the difference. Experiment with different modes: if you select `mode = 'constant'`, you can add the parameter `cval` (scalar), to specify the value of the pixels outside of the image borders. You can also try to set the parameter `reshape = False` to see the result.


```sos
%use sos

# Apply operation
rot_car_ndi = ndi.interpolation.rotate(car, angle = 45, reshape=True, mode='reflect')
# Compare to original
plt.close('all')
cv_rotate_viewer = viewer([car, rot_car_ndi], title = ['Original', 'Rotated'], subplots=[1,2])
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


### Edge Detection in SciPy NDImage

Again, we will use the Sobel filter. The basic syntax in `ndi` is  (see [the documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html)):
```python
output = scipy.ndimage.sobel(input, axis= -1, mode='reflect')
```

The parameters are:
* `input` (`numpy` array): The source image, 
* `axis` (floating point number): axis through which to apply the 1-dimensional Sobel filter (defaults to -1, i.e., the last axis, i.e., the 2nd (1) for grayscale images),
* `mode` (string): Specify the boundary conditions (see the last example).

Again, we can see differences between the two libraries. In OpenCV you could specify different orders in each direction. In `scipy.ndimage`, the order is always 1, and you can only specify one direction. So to get an image like the one resulting from OpenCV's function, you would need to call `ndi`s function twice (once with `axis = 0` and once with `axis = 1`), and then get the norm of the gradient ($\sqrt{[\partial f/\partial x]^2 + [\partial f / \partial y]^2}$, where $f$ represents the image). Do not worry if this is not clear to you at this point, it will become much clearer as you go through the course and study directional derivatives and the Sobel filter. 

Run the next cell to see the Sobel filter applied to the image `hands`. Change the argument axis to see the effect.


```sos
%use sos

#Apply filter
hands_edge = ndi.sobel(hands, axis= 0, mode='reflect')
# Compare
plt.close('all')
ndi_sobel_viewer = viewer([hands, hands_edge], title = ['Original', 'Edges'], subplots=[1,2])
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


### Rotation in Scikit Image

The basic syntax of the function for grayscale images is  (see [the documentation](https://scikit-image.org/docs/stable/api/skimage.transform.html?highlight=rotate#skimage.transform.rotate)):
```python
output = skimage.transform.rotate(image, angle, resize=False, center=None, mode='reflect', preserve_range = True)
```

The parameters are:
* `input` (`numpy` array): The source image, 
* `angle` (floating point number): Rotation angle in degrees,
* `resize` (boolean): Specify whether to resize the image to contain the full original. Defaults to `False`,
* `center` (tuple of length 2): Specifies the center of rotation. The order is (columns, rows),
* `mode` (string): Specify boundary conditions (see the last example). 

In Scikit Image, we have a new parameter that nor OpenCV nor Scipy NDImage have. Scikit Image lets you specify the center of rotation, which for some applications can come as a very handy tool.

We will apply Scikit Image's rotation to `car` again, this time for an angle of $60^\circ$. Moreover, we will change the center of rotation. Run the next cell to see the result. Modify the cell and experiment with the different parameters.


```sos
%use sos

# First, we import the module transform
from skimage import transform

# Now we apply operation
rot_car_skimage = skimage.transform.rotate(car, angle = 60, resize=False, center=(50, 50), mode='reflect', preserve_range = True)

# And visualize
plt.close('all')
skimage_rotate_viewer = viewer([car, rot_car_skimage], title = ['Original', 'Rotated'], subplots=[1,2])
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


### Edge Detection in Scikit Image

The basic syntax of the Sobel filter for grayscale images is  (see [the documentation](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel)):

```python
output = skimage.filters.sobel(image, axis=None, mode='reflect')
```

The parameters are:
* `image` (`numpy` array): The source image, 
* `axis` (integer): If given, compute the axis in the specified direction. If `None`, the magnitude of the gradient is calculated,
* `mode` (string): Specify boundary conditions (see example above). 

As you can see, in this function, Scikit image gives more freedom to the user (you can produce both the magnitude of the gradient, and the gradient in each individual axis). Moreover, if you need a quicker and simpler way, Scikit-image has the filters `sobel_h` and `sobel_v`, that without any extra parameters apply the corresponding (horizontal or vertical) filter.

Run the next cell to apply Scikit-Image Sobel filter on the image `plate`. Modify the cell and play with the different values and with different images.


```sos
%use sos

# First, we import the module filters
from skimage import filters

# Apply filter
plate_edges_skimage = skimage.filters.sobel(plate, axis=None, mode='reflect')

# Visualize
plt.close('all')
skimage_edge_viewer = viewer([plate, plate_edges_skimage], title = ['Original', 'Edges'], subplots=[1,2])
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


As you have realized by this brief introduction, no library is perfect, entirely complete, nor generally better than the others. For some applications and for some functions, you will find one library to be more useful than the others. In some cases, some libraries will not have the functions you need! So in complex applications, you may need to use two of them, or even combine them all. In general, it is good to be aware of your options. 

## <a class="anchor"></a> 5.B. Try it yourself! (2 points)

Congratulations! You are almost at the end of the introductory lab, this is your last task. It is your time to explore the three libraries (explore documentation, tutorials, etc.). A good access point is to go to their website. For **2 points**, then, choose one of the libraries and implement a **Gaussian smoothing (a.k.a. Gaussian blur)** on the image `boats`. Modify the cell below, and store your result in the variable `blurred_boats`. Use:
* $\sigma = 5$,
* `'reflect'` (or `cv.BORDER_REFLECT`) boundary conditions, 
* and use the default kernel size. 

If applicable, use: 
* `preserve_range = True`, and/or 
* `ksize = (0, 0)`.


```sos
%use sos

blurred_boats = np.copy(boats)
blurred_boats = cv.GaussianBlur(blurred_boats,(0,0),cv.BORDER_DEFAULT)
```


```sos
%use sos
# Visualize 
plt.close('all')
gaussian_viewer = viewer([boats, blurred_boats], title = ['Original', 'Blurred'], subplots=[1,2]) 
```


    HBox(children=(Output(layout=Layout(width='80%')), Output(), Output(layout=Layout(width='25%'))))



    Button(description='Show Widgets', style=ButtonStyle())


<p><b>Congratulations on finishing Lab 0!</b></p>
<p>
Make sure to save your notebook (keep a copy on your personal computer for reference) and upload it to <a href="https://moodle.epfl.ch/course/view.php?id=522">Moodle</a>.
</p>

<div class="alert alert-danger">
<h4>Feedback</h4>
    <p style="margin:4px;">
    This is the first edition of the image-processing laboratories using Jupyter Notebooks running on Noto. Do not leave before giving us your <a href="https://moodle.epfl.ch/mod/feedback/view.php?id=1091265">feedback here!</a></p>
</div>
