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

Matplotlib is used for plotting, plots are directly embedded in the notebook thanks to the '%matplolib inline' command at the beginning. We have that `np.random.randn(d0, d1...)` returns a sample from the normal distribution where the d0, d1 etc parameters are the dimensions of the array that we are returning 
