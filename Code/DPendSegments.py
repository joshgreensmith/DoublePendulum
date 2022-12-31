import math
import time
import numpy as np
from scipy.integrate import odeint
from numba import jit
import colorsys
import matplotlib.pyplot as plt
import argparse

start_time = time.time()

# Parse in arguments for simulation through terminal (uncomment Tmax to change maximum time for integration from 100)
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mass", help="Change the mass of each pendulum", type=int)
parser.add_argument("-l", "--length", help="Change the length of each pendulum", type=int)
parser.add_argument("-res", "--resolution", help="Change the gridSize", type=int)
# parser.add_argument("-t", "--Tmax", help="Change the maximum time for each grid to run for", type=int)

args = parser.parse_args()

gridSize = args.resolution
m = args.mass
l = args.length
# Tmax = args.Tmax
Tmax = 100

g = 9.81
TStep = 10

Tnumber = Tmax + 1
pi = math.pi

# Define the differential equations for the double pendulum system and use the JIT compiler to spped up the process
@jit
def doublePendODE(y, t, l, m, g):
    # Define the differential equations
    theta1, theta2, p1, p2 = y

    dtheta1 = (6 * (2 * p1 - 3 * p2 * math.cos(theta1 - theta2))) / (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))
    dtheta2 = (6 * (2 * p2 - 3 * p1 * math.cos(theta1 - theta2))) / (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))

    dp1 = - 0.5 * m * l**2 * (dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))
    dp2 = - 0.5 * m * l**2 * (-dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))

    dydx = [dtheta1, dtheta2, dp1, dp2]
    return dydx

# Define the flip time testing function and use the numpy.vectorize addon to allow multiple computations at the same time in the colour map
@np.vectorize
def doublePendTestSegments(theta1, theta2, l):
    y0 = [theta1, theta2, 0.0, 0.0]

    for t in range(Tmax):
        sol = odeint(doublePendODE, y0, [0,1], args=(l, m, g), mxstep=5000000)

        prevTheta2 = y0[1]  % (2 * pi)
        Theta2 = sol[1][1] % (2 * pi)

        if prevTheta2 < pi and prevTheta2 > pi - 0.52 and Theta2 > pi:
            return t + 1

        if prevTheta2 > pi and prevTheta2 < pi + 0.52 and Theta2 < pi:
            return t + 1

        y0 = sol[1]
        nextTheta2 = y0[1]

    return Tmax


# Define mesh space for the colour map
x = np.linspace(-3, 3, gridSize)
y = np.linspace(-3, 3, gridSize)
X,Y = np.meshgrid(x,y)
Z = doublePendTestSegments(X, Y, length)

# Set up 2D plot using matplotlib
fig = plt.figure(figsize = (12,10))
fig.subplots_adjust(wspace=0.5)

# Set up latex usage and font to match final write up
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('font', family='cooper')

# Create colour map
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('plasma'))
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('Time to flip (s)', fontsize = 14)

# Label the axes and fix the aspect ratio to make the colour map a square
plt.xlabel(r'$\boldsymbol{\theta_{1}} $ (Radians)', fontsize = 14)
plt.ylabel(r'$\boldsymbol{\theta_{2}} $ (Radians)', fontsize = 14)
plt.axis([-3, 3, -3, 3])
plt.axes().set_aspect('equal')

# Save the colour map as a PNG with the parsed arguments in the name
name = "length" + str(length)  + "_mass" + str(m) + "_res" + str(gridSize) + ".png"
plt.savefig(name, bbox_inches='tight', dpi = 500)

# Print the time to run the simulation taking seconds if it is less than a minute
secondsToRun = time.time() - start_time
if secondsToRun < 60:
    print("--- %s seconds ---" % (math.ceil(secondsToRun*1000)/1000))
else:
    print("Time to run = " + str(time.strftime("%H:%M:%S", time.gmtime(secondsToRun))))