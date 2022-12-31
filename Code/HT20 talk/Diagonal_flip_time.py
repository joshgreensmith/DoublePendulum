from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit

m = 1.0
l = 50.0
g = 9.81
Tmax = 100
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

thetaRange = np.linspace(start = -2, stop = 2, num = 30000)

flipTimeOutput = []

for angle in thetaRange:
    print(angle)
    flipTimeOutput.append(doublePendTest(angle, angle))

plt.plot(thetaRange, flipTimeOutput)
plt.xlabel('Theta1, Theta2')
plt.ylabel('Flip time')
plt.title('Flip time across the diagonal')
file_name = 'Diagonal_flip_time_m_' + str(round(m)) + '_l_' + str(round(l)) + '.png'
plt.savefig(file_name)