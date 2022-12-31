
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math

# First method for solving without segments
def doublePend(y, t, l, m, g):
    theta1, theta2, p1, p2 = y
    dtheta1 = (6*(2*p1 - 3 * p2 * math.cos(theta1 - theta2)))/(m*l**2*(16-9*(math.cos(theta1-theta2)**2)))
    dtheta2 = (6*(2*p2 - 3 * p1 * math.cos(theta1 - theta2)))/(m*l**2*(16-9*(math.cos(theta1-theta2)**2)))

    dp1 = - 0.5 * m * l**2 * (dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))
    dp2 = - 0.5 * m * l**2 * (-dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))

    dydx = [dtheta1, dtheta2, dp1, dp2]
    return dydx


m = 10
l = 100
theta1 = 0.3333333333333
theta2 = 0.0
g = 9.81
Tmax = 100

y0 = [theta1, theta2, 0.0, 0.0]
t = np.linspace(0, Tmax, Tmax + 1)
sol = odeint(doublePend, y0, t, args=(l, m, g))


# Second method for solving with segments to improve performance
TStep = 10
newSol = []
t0Sol = [theta1, theta2, 0.0, 0.0]
for j in range(0, Tmax, TStep):
        newT = np.linspace(0, TStep, TStep + 1)
        segmentSol = odeint(doublePend, t0Sol, newT, args=(l, m, g), mxstep=5000000)
        #newSol.append(segmentSol)
        t0Sol = segmentSol[TStep]
        if j > 0:
            for i in range(1, len(segmentSol)):
                newSol.append(segmentSol[i][0])
        else:
            for i in range(0, len(segmentSol)):
                newSol.append(segmentSol[i][0])


plt.plot(sol[:, 0], 'b', label='Theta 1 without Segments')
plt.plot(newSol, 'g', label='Theta 1 with Segments')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Angle (Radians)')
plt.grid()
plt.show()

