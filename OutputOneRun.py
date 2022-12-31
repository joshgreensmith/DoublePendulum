from numba import jit
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

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


y0 = [0.1, pi/2, 0.0, 0.0]
l = 50
m = 1
g = 9.81
dPendSol = np.array(odeint(doublePend, y0, list(np.arange(0.0, 1000.0, 1.0)), args=(l,m,g), mxstep=5000000))

plt.plot(dPendSol[:, 0])
plt.plot(dPendSol[:, 1])
plt.show()