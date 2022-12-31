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





Tmax = 100
Tnumber = Tmax + 1

l = 1
m = 1
g = 9.81

pi = np.pi

errorCount = 0

rangeValues = np.linspace(0, 3, 10)
for theta1 in rangeValues:
    for theta2 in rangeValues:
        fliptime = doublePendTest(theta1, theta2)
        fliptimeSegments = doublePendTestSegments(theta1, theta2)

        #print("Without segments = " + str(fliptime) + " \n"  + "With segments = " + str(fliptimeSegments) + "\n")
        if fliptime != fliptimeSegments:
            print(theta1, theta2, fliptime, fliptimeSegments)
            errorCount += 1


print("ErrorCount = " + str(errorCount) + "/" + str(len(rangeValues)**2))

