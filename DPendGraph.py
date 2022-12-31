from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math


def doublePend(y, t, l, m, g):
    theta1, theta2, p1, p2 = y
    # mu = 1 + (m1 / m2)
    dtheta1 = (6 * (2 * p1 - 3 * p2 * math.cos(theta1 - theta2))) / \
        (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))
    dtheta2 = (6 * (2 * p2 - 3 * p1 * math.cos(theta1 - theta2))) / \
        (m * l**2 * (16 - 9 * (math.cos(theta1 - theta2)**2)))

    dp1 = - 0.5 * m * l**2 * \
        (dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3 * (g / l) * math.sin(theta1))
    dp2 = - 0.5 * m * l**2 * \
        (-dtheta1 * dtheta2 * math.sin(theta1 - theta2) +
         3 * (g / l) * math.sin(theta1))

    dydx = [dtheta1, dtheta2, dp1, dp2]
    return dydx


m = 20
l = 80
g = 9.81

Tmax = 200
Tnumber = Tmax * 2 + 1

theta1 = 0.1
theta2 = 0.4

y0 = [theta1, theta2, 0.0, 0.0]
t = np.linspace(0, Tmax, Tnumber)
sol = odeint(doublePend, y0, t, args=(l, m, g))

# Set up latex usage and font to match final write up
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('font', family='cooper')

plt.plot(t, sol[:, 0], 'b', label=r'$\boldsymbol{\theta_{1}}$')
plt.plot(t, sol[:, 1], 'g', label=r'$\boldsymbol{\theta_{2}}$')
plt.legend(loc='best')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (Radians)')
plt.grid()

name = "ODEGraph_length" + str(l)  + "_mass" + str(m) + "_theta1_" +str(math.ceil(theta1*100)/100) + "_theta2_" + str(math.ceil(theta2*100)/100) + ".png"
plt.savefig(name, bbox_inches='tight', dpi = 500)