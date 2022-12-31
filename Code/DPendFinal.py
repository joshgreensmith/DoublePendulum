import math
import time
import numpy as np
from scipy.integrate import odeint
from numba import jit
import colorsys
import matplotlib.pyplot as plt
import argparse

start_time = time.time()

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mass", help="Change the mass of each pendulum", type=int)
parser.add_argument("-l", "--length", help="Change the length of each pendulum", type=int)
parser.add_argument("-res", "--resolution", help="Change the gridSize", type=int)
parser.add_argument("-t", "--Tmax", help="Change the maximum time for each grid to run for", type=int)

args = parser.parse_args()

gridSize = args.resolution
m = args.mass
l = args.length
g = 9.81

Tmax = args.Tmax
Tnumber = Tmax + 1

@jit
def doublePend(y, t, l, m, g):
    # Define the differential equations
    theta1, theta2, p1, p2 = y

    dtheta1 = (6 * (2 * p1 - 3 * p2 * math.cos(theta1 - theta2))) / (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))
    dtheta2 = (6 * (2 * p2 - 3 * p1 * math.cos(theta1 - theta2))) / (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))

    dp1 = - 0.5 * m * l**2 * (dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))
    dp2 = - 0.5 * m * l**2 * (-dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))

    dydx = [dtheta1, dtheta2, dp1, dp2]
    return dydx

@np.vectorize
def doublePendTest(theta1, theta2):
    # Define starting values
    y0 = [theta1, theta2, 0.0, 0.0]
    t = np.linspace(0, Tmax, Tnumber)
    sol = odeint(doublePend, y0, t, args=(l, m, g), mxstep=5000000)

    # Extract theta2 from the set of solutions
    theta2Solution = []
    for i in sol:
        theta2Solution.append(i[1])

    # Test whether theta2 has crossed the boundary conditions and flipped in the time
    # Convert this to use radians instead of loose conversion into degrees and then back into radians
    for i in range(len(theta2Solution)):
        if i > 0:
            prevTheta2 = theta2Solution[i - 1]
            Theta2 = theta2Solution[i]

            prevRotatedPendulum2Bearing = (prevTheta2 * 360 / (2 * math.pi)) % 360
            rotatedPendulum2Bearing = (Theta2 * 360 / (2 * math.pi)) % 360

            if prevRotatedPendulum2Bearing < 180 and prevRotatedPendulum2Bearing > 150 and rotatedPendulum2Bearing > 180:
                return t[i]

            if prevRotatedPendulum2Bearing > 180 and prevRotatedPendulum2Bearing < 210 and rotatedPendulum2Bearing < 180:
                return t[i]
    return Tmax+1




# Define mesh space
x = np.linspace(-3, 3, gridSize)
y = np.linspace(-3, 3, gridSize)
X,Y = np.meshgrid(x,y)
Z = doublePendTest(X,Y)

# Set up 2d plot
fig = plt.figure(figsize = (12,10))
fig.subplots_adjust(wspace=0.3)

# Set up colourmap and colourbar
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('plasma'))
cbar = plt.colorbar()
cbar.set_label('Time to flip (s)')
plt.xlabel('Theta1')
plt.ylabel('Theta2')

plt.axis([-3, 3, -3, 3])
name = "length" + str(l)  + "_mass" + str(m) + "_res" + str(gridSize) + "_tmax" + str(Tmax) + ".png"
plt.savefig(name, bbox_inches='tight')

secondsToRun = time.time() - start_time
if secondsToRun < 60:
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    print("Time to run = " + str(time.strftime("%H:%M:%S", time.gmtime(secondsToRun))))
