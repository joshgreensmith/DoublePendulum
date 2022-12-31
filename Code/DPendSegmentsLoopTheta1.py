import math
import time
import numpy as np
from scipy.integrate import odeint
import colorsys
import matplotlib.pyplot as plt
from numba import jit

# Define the differential equations for the double pendulum system and use the JIT compiler to spped up the process
@jit
def doublePendODE(y, t, l, m, g):
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

        prevTheta1 = y0[0]  % (2 * pi)
        Theta1 = sol[1][0] % (2 * pi)

        if prevTheta1 < pi and prevTheta1 > pi - 0.52 and Theta1 > pi:
            return t + 1

        if prevTheta1 > pi and prevTheta1 < pi + 0.52 and Theta1 < pi:
            return t + 1

        y0 = sol[1]

    return Tmax

# Define the overarching variables across the colour maps
gridSize = 50
start_time = time.time()

m = 1
g = 9.81

Tmax = 100
Tnumber = 101

pi = math.pi

# Define the lengths to test in this run
# lengthsToTest = [1, 100, 300, 500]
lengthsToTest = [1]

# Define the image output function for each length to test in the array
def imageOutput(length):
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
    name = "length" + str(length)  + "_mass" + str(m) + "_res" + str(gridSize) + "_theta1" + ".png"
    plt.savefig(name, bbox_inches='tight', dpi = 500)

    # Print the time to run the simulation taking seconds if it is less than a minute
    secondsToRun = time.time() - start_time
    if secondsToRun < 60:
        print("Time to run for l = " + str(length) + " is " + str(math.ceil(secondsToRun*100)/100) + " seconds")
    else:
        print("Time to run for l = " + str(length) + " is " + str(time.strftime("%H:%M:%S", time.gmtime(secondsToRun))))

# Run the simulation on all the chosen lengths
for lengthToTest in lengthsToTest:
    print("Testing length = " + str(lengthToTest))
    imageOutput(lengthToTest)

