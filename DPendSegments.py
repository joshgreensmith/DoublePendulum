# Code written by Josh Greensmith for High Master Summer Essay on "The Hidden Beauty of the Double Pendulum"

# Import all required modules
import math
import time
import numpy as np
from scipy.integrate import odeint
from numba import jit
import matplotlib.pyplot as plt
import argparse

# Start program timer to see how long it takes to run
start_time = time.time()

# Parse in specific starting conditions for colour map using the argparse library and define all variables based on these
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mass", help = "Change the mass of each pendulum", type = int, default = 1)
parser.add_argument("-l", "--length", help = "Change the length of each pendulum", type = int, default = 1)
parser.add_argument("-res", "--resolution", help = "Change the gridSize", type = int, default = 10)
parser.add_argument("-t", "--Tmax", help = "Change the maximum time for each grid to run for", type = int, default = 100)

args = parser.parse_args()

gridSize = args.resolution
m = args.mass
l = args.length
Tmax = args.Tmax

# General global parameters
g = 9.81
TStep = 10
dpi = 300

Tnumber = Tmax + 1
pi = math.pi

# These are the core differential equations of the system in a function that returns the derivatives of the characteristic equations given input points
# The @jit is used to compile the python into lower level C code that runs far faster in repeated loops
@jit
def doublePend(y, t, l, m, g):
    # Differential equations
    theta1, theta2, p1, p2 = y

    dtheta1 = (6 * (2 * p1 - 3 * p2 * math.cos(theta1 - theta2))) / (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))
    dtheta2 = (6 * (2 * p2 - 3 * p1 * math.cos(theta1 - theta2))) / (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))

    dp1 = - 0.5 * m * l**2 * (dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))
    dp2 = - 0.5 * m * l**2 * (-dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))

    dydx = [dtheta1, dtheta2, dp1, dp2]
    return dydx

# Function for cutting run time -> when a flip is detected it returns the time immediately, otherwise it returns Tmax -> segmentation
# The @np.vectorize allows several functions to be run at the same time which speeds up runtime significantly
@np.vectorize
def doublePendTestSegments(theta1, theta2):
    y0 = [theta1, theta2, 0.0, 0.0]

    for t in range(Tmax):
        sol = odeint(doublePend, y0, [0,1], args=(l, m, g), mxstep=5000000)

        prevTheta2 = y0[1]  % (2 * pi)
        Theta2 = sol[1][1] % (2 * pi)

        if prevTheta2 < pi and prevTheta2 > pi - 0.52 and Theta2 > pi:
            return t + 1

        if prevTheta2 > pi and prevTheta2 < pi + 0.52 and Theta2 < pi:
            return t + 1

        y0 = sol[1]
        nextTheta2 = y0[1]

    return Tmax + 1

# The rest of the code is just for formatting the colourmap in Matplotlib
# This took the most time despite being the "easiest" part of solving the differential equations, displaying the data in a nice way is very time consuming

# Define mesh space for the colour map
x = np.linspace(-3, 3, gridSize)
y = np.linspace(-3, 3, gridSize)
X,Y = np.meshgrid(x,y)
Z = doublePendTestSegments(X, Y)

# Set up 2D plot using matplotlib
fig = plt.figure(figsize = (12,10))
fig.subplots_adjust(wspace=0.5)

# Set up latex usage and Cooper font
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('font', family='cooper')

# Create colour map using colormesh and colorbar
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('plasma'), vmin = 0, vmax = 100)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('Time to flip (s)', fontsize = 14)

# Label the axes and fix the aspect ratio to make the colour map a square
plt.xlabel(r'$\boldsymbol{\theta_{1}} $ (Radians)', fontsize = 14)
plt.ylabel(r'$\boldsymbol{\theta_{2}} $ (Radians)', fontsize = 14)
plt.axis([-3, 3, -3, 3])
plt.axes().set_aspect('equal')

# Save the colour map as a PNG with the parsed arguments in the name
name = "length" + str(l)  + "_mass" + str(m) + "_res" + str(gridSize) + ".png"
plt.savefig(name, bbox_inches='tight', dpi = dpi)

# Print the time to run the simulation taking seconds if it is less than a minute
secondsToRun = time.time() - start_time
if secondsToRun < 60:
    print("Time to run for l = " + str(l) + " is " + str(math.ceil(secondsToRun*100)/100) + " seconds")
else:
     print("Time to run for l = " + str(l) + " is " + str(time.strftime("%H:%M:%S", time.gmtime(secondsToRun))))
