<img src="chameleon_logo.png" alt="drawing" width="200"/>

# CHAMELEON
CHAMELEON is an open-source Python package to accommodate model calibration and Gaussian process emulation (GPE).


## Installation
### Virtual environment
It is recommended to install CHAMELEON in a virtual environment. To create a virtual environment using Conda, first install a distribution of [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/miniconda.html). After installation, open a terminal window to create a new virtual environment named (for example) `monarch` that uses `Python 3.12`:
```
conda create -n chameleon python=3.12
conda activate chameleon
```
### Installing using pip (not yet available)

CHAMELEON can be installed using pip:
```
pip install chameleon-beatlab
```
### Installing from source
Download the latest source code from Github. From a terminal window in the directory where you want to download and install the MONARCH source code, run:
```
git clone https://github.com/BeatLabUCI/chameleon.git
```
Then build and install CHAMELEON from the top level of the source tree:
```
pip install .
```

### Checking your installation
To check that CHAMELEON has been installed correctly, open a Python interpreter (with the correct virtual environment if applicable) and run:
```
import chameleon
```

## Using CHAMELEON
We include several Jupyter notebooks to demonstrate how to use CHAMELEON. These notebooks can be found in the `examples` directory.