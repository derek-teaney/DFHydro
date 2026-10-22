# vischydro module for setting up and running the VischydroMain.exe code
""" Vischydro module for setting up and running the VischydroMain.exe code

This module provides functions to set the input parameters neeeded to run the VischydoroMain.exe code, create initial data grids, and run the simulation. 

The user simply imports the vischydro module, modifies the input_data dictionary as needed, creates an initial data grid, and then calls the runcode function to execute the simulation.  

Example usage:

import vischydro

# Modify the input parameters
vischydro.input_data["Vischydro/nx"] = 200

# Create an initial data grid this will be a numpy array of shape (ny, nx, ndof)
initialdata = vischydro.initialdata_grid(vischydro.input_data)

# Fill in the initialdata array as needed...

# Run the vischydro simulation
vischydro.runcode(initialdata, vischydro.input_data, runcommand='./VischydroMain.exe')

# This launches the vischydro code with the specified initial data and input parameters.
"""

import json
from collections import UserDict
import subprocess
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import pdb



_input_data = {
    "Vischydro": {
        # These are defaults and should not normally be changed
        "cfl_max": 0.8,
        "is_bjorken": True,
        "highest_order_term_only": True, 
        "use_ideal_step_only": False,
        "is_periodic": False
    },
    "VischydroMain": {
        "ndof": 15,
        "nx": 80,
        "ny": 90,
        "xmin": -8.0,
        "xmax": 8.0,
        "ymin": -9.0,
        "ymax": 9.0,
        "print_frequency": 10,
        "t_start": 0.0,
        "t_end": 47.0,
        "dt_max": 1.0,
        "run_name": "DFTest",
        "initial_conditions_filename": "DFTest_ic.h5",
        # possible values are "primitives" or "charges"
        "initial_field_type" : "primitives",
        "eta_by_s": 1./(4.*np.pi),
        "zeta_by_s": 0.0
    }
}

class FixedData(UserDict):
    """ This is a dictionary helper classes that does not allow adding new keys,
    preventing typos. 

    It also allows accessing nested dictionary elements using
    a single keystring with '/' separators. For example, to access the value of
    key 'key3' in nested dictionary {'key1': {'key2': {'key3': value}}}, one can
    use the keystring mydict['key1/key2/key3'].  The class checks that each key
    exists and does not allow adding new keys. """

    def __init__(self, initialdata):
        # super().__init__(initialdata)
        self.data = initialdata

    def __setitem__(self, key, value):
        """ Navigate the nested dictionary using the provided keystring
        checking that each key exists before setting the value to v. The format
        of keystring is mydict['key1/key2/key3'] for nested dictionaries. """
        keystring = key.split('/')
        name = keystring[-1]
        d = self.data
        for i, key in enumerate(keystring[:-1]):
            d = d[key]
        if d.get(name) is None:
            raise KeyError("Key '{}' not found in dictionary.".format(name))
        d[name] = value

    def __getitem__(self, key):
        """ Navigate the nested dictionary using the provided keystring
        checking that each key exists before getting the value to v. The format
        of keystring is 'key1/key2/key3' for nested dictionaries. """
        keystring = key.split('/')
        name = keystring[-1]
        d = self.data
        for i, key in enumerate(keystring[:-1]):
            d = d[key]
        if d.get(name) is None:
            print(keystring)
            raise KeyError("Key '{}' not found in dictionary.".format(name))
        return d[name]


input_data = FixedData(_input_data)
""" A dictionary holding the default input parameters for the vischydro code.
This can be modified by the user as needed before running the simulation. This
is just a nested dictionary. 

Example usage:

import vischydro
vischydro.input_data["VischydroMain/nx"] = 200

print(vischydro.input_data) 
"""


# Create a Vischydro of xy values based on the input_data dictionary


def xygrid(input_data):
    """ Create the Vischydro coordinate grid based on the input data. 

    Returns:
    xarray, yarray, dx, dy : np.ndarray, np.ndarray, float, float
        The one dimensional x and y coordinate arrays, and the grid spacings dx and dy.
    """
    
    xmin = input_data['VischydroMain']['xmin']
    xmax = input_data['VischydroMain']['xmax']
    ymin = input_data['VischydroMain']['ymin']
    ymax = input_data['VischydroMain']['ymax']
    NX = input_data['VischydroMain']['nx']
    xarray = np.linspace(xmin, xmax, NX, endpoint=False)
    NY = input_data['VischydroMain']['ny']
    yarray = np.linspace(ymin, ymax, NY, endpoint=False)
    dx = xarray[1] - xarray[0]
    dy = yarray[1] - yarray[0]
    # create a meshgrid
    xarray, yarray = np.meshgrid(xarray, yarray, indexing='xy')
    print("xygrid: ", "Created Vischydro coordinate grid based on the input data with nx =", NX, " ny =", NY)
    return xarray, yarray, dx, dy


def initialdata_grid(input_data):
    """ Create an initialdata grid based on the input data. The initialdata grid is a numpy array of shape (ny, nx, ndof) filled with zeros."""
    
    nx = input_data['VischydroMain']['nx']
    ny = input_data['VischydroMain']['ny']
    ndof = input_data['VischydroMain']['ndof']
    initialdata = np.zeros(
        (input_data['VischydroMain']['ny'], input_data['VischydroMain']['nx'], input_data['VischydroMain']['ndof']))
    print("initialdata_grid:", "Created Vischydro coordinate grid based on the input data with nx =",
          nx, " ny =", ny, " ndof =", ndof)
    return initialdata


def timegrid(input_data):
    """Predict the sequence of output times written by the simulation.

    Uses cfl_max * min(dx, dy) as the internal time step (capped by dt_max),
    then returns every print_frequency-th step from t_start to t_end.
    This is an estimate — the actual solver may adapt dt within a step.
    """
    xmin = input_data['VischydroMain/xmin']
    xmax = input_data['VischydroMain/xmax']
    ymin = input_data['VischydroMain/ymin']
    ymax = input_data['VischydroMain/ymax']
    nx   = input_data['VischydroMain/nx']
    ny   = input_data['VischydroMain/ny']
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    cfl    = input_data['Vischydro/cfl_max']
    dt_max = input_data['VischydroMain/dt_max']
    dt     = min(cfl * min(dx, dy), dt_max)

    t_start = input_data['VischydroMain/t_start']
    t_end   = input_data['VischydroMain/t_end']
    pfreq   = input_data['VischydroMain/print_frequency']

    dt_out = pfreq * dt
    n_out  = int((t_end - t_start) / dt_out) + 1
    iout   = np.arange(n_out)
    times  = t_start + iout * dt_out
    mask   = times <= t_end + 1e-12 * dt_out
    iout, times = iout[mask], times[mask]
    return {'t': times, 'step': iout * pfreq, 'iout': iout}


def set_timegrid(input_data, dt_out):
    """Set print_frequency and dt_max so outputs occur every dt_out in physical time.

    Computes the natural CFL time step dt, then finds the smallest print_frequency
    such that print_frequency * dt_max = dt_out with dt_max <= dt.
    Modifies input_data in-place.
    """
    xmin = input_data['VischydroMain/xmin']
    xmax = input_data['VischydroMain/xmax']
    ymin = input_data['VischydroMain/ymin']
    ymax = input_data['VischydroMain/ymax']
    nx   = input_data['VischydroMain/nx']
    ny   = input_data['VischydroMain/ny']
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    cfl    = input_data['Vischydro/cfl_max']
    dt_max = input_data['VischydroMain/dt_max']
    dt = min(cfl * min(dx, dy), dt_max)

    print_frequency = int(dt_out / dt) + 1
    dt_max_new = dt_out / print_frequency

    input_data['VischydroMain/print_frequency'] = print_frequency
    input_data['VischydroMain/dt_max'] = dt_max_new
    print("set_timegrid: dt={:.4g}, print_frequency={}, dt_max={:.4g}, dt_out={:.4g}".format(
        dt, print_frequency, dt_max_new, dt_out))


# one shoudl keep these options unless you know what you are doing
options = {}

# potential options
#    '-ts_type': 'eimex',
#    '-ts_max_snes_failures': '5',
#    '-pc_type': 'jacobi',
#    '-ksp_type': 'bcgs',
#    '-ts_adapt_type': 'none',

def runcode(initialdata, input_data, runcommand='./VischydroMain.exe', actually_run=True, petsc_args=''):
    """ Run VicschydroMain.exe

    Dump the current input input_data to a json file (default name is
    'RUNNAME.json'), which serves as input to the vischydro code.
    Additional arguments can be passed to the code on the command
    line by adding them to the options input_data structure, see option.
    
    The initialdata array is written to an HDF5 file with filename
    input_data['icfilename'], creating an array initialdata. This is then
    read by the vischydro code. """

    # Create an HDF5 file
    icfilename = input_data['VischydroMain/run_name'] + '_ic.h5'

    input_data['VischydroMain/initial_conditions_filename'] = icfilename
    with h5py.File(icfilename, 'w') as file:
        # Create a dataset in the file and write the array to it
        file.create_dataset('initialdata', data=initialdata)

    inputs = input_data['VischydroMain/run_name'] + '.json'
    with open(inputs, 'w') as file:
        json.dump(input_data.data, file, indent=4)

    opts = []
    for kev, val in options.items():
        if val is None or val == '':
            opts += [kev]
        else:
            opts += [kev, val]
    if petsc_args:
        opts += petsc_args.split()

    runcommand_array = runcommand.split()
    command = runcommand_array + ['-input', inputs] + opts[:]
    print("Executing command: ")
    print(" ".join(command))
    if actually_run:
        subprocess.call(command)


# Swap the Vischydro dimensions for x and y
def swap_xy(input_data):
    # swap nx and ny
    nx = input_data['VischydroMain']['nx']
    ny = input_data['VischydroMain']['ny']
    input_data['VischydroMain/nx'] = ny
    input_data['VischydroMain/ny'] = nx
    # swap xmin and ymin
    xmin = input_data['VischydroMain']['xmin']
    ymin = input_data['VischydroMain']['ymin']
    input_data['VischydroMain/xmin'] = ymin
    input_data['VischydroMain/ymin'] = xmin
    # swap xmax and ymax
    xmax = input_data['VischydroMain']['xmax']
    ymax = input_data['VischydroMain']['ymax']
    input_data['VischydroMain/xmax'] = ymax
    input_data['VischydroMain/ymax'] = xmax