import math
import time
import numpy as np
from scipy.integrate import odeint
from numba import jit
import colorsys
import matplotlib.pyplot as plt
import argparse

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
    for i in range(len(theta2Solution)):
        if i > 0:
            prevTheta2 = theta2Solution[i - 1]  % (2 * pi)
            Theta2 = theta2Solution[i] % (2 * pi)

            if prevTheta2 < pi and prevTheta2 > pi - 0.52 and Theta2 > pi:
                return t[i]

            if prevTheta2 > pi and prevTheta2 < pi + 0.52 and Theta2 < pi:
                return t[i]

    # If no flip has occured return the max time + 1
    return Tmax+1


@np.vectorize
def doublePendTestSegments(theta1, theta2):

    t0Sol = [theta1, theta2, 0.0, 0.0]
    t = np.linspace(0, TStep, TStep + 1)
    # Start loop for t increments and testing to increase performance
    for j in range(0, Tmax, TStep):
        #print(j)

        segmentSol = odeint(doublePend, t0Sol, t, args=(l, m, g), mxstep=5000000)

        theta2Solution = []
        # if j > 0:
        #     for i in range(1, len(segmentSol)):
        #         theta2Solution.append(segmentSol[i][1])
        # else:
            # for i in range(0, len(segmentSol)):
            #     theta2Solution.append(segmentSol[i][1])

        for i in range(len(segmentSol)):
                theta2Solution.append(segmentSol[i][1])
        
        # Test whether theta2 has crossed the boundary conditions and flipped in the time
        for i in range(len(theta2Solution)):
            if i > 0:
                prevTheta2 = theta2Solution[i - 1]  % (2 * pi)
                Theta2 = theta2Solution[i] % (2 * pi)
            else:
                prevTheta2 = t0Sol[1] % (2 * pi)
                Theta2 = theta2Solution[0] % (2 * pi)

            if prevTheta2 < pi and prevTheta2 > pi - 0.52 and Theta2 > pi:
                return j + t[i]

            if prevTheta2 > pi and prevTheta2 < pi + 0.52 and Theta2 < pi:
                return j + t[i]

        t0Sol = segmentSol[TStep]

    # If no flip has occured return the max time + 1
    return Tmax+1

Tmax = 100
Tnumber = 101

TStep = 10

l = 1
m = 1
g = 9.81

pi = np.pi

errorCount = 0

rangeValues = np.linspace(0, 3, 20)
for theta1 in rangeValues:
    #print(theta1)
    for theta2 in rangeValues:
        fliptime = doublePendTest(theta1, theta2)
        fliptimeSegments = doublePendTestSegments(theta1, theta2)

        #print("Without segments = " + str(fliptime) + " \n"  + "With segments = " + str(fliptimeSegments) + "\n")
        if fliptime != fliptimeSegments:
            print(theta1, theta2, fliptime, fliptimeSegments)
            errorCount += 1


print("ErrorCount = " + str(errorCount) + "/" + str(len(rangeValues)**2))
