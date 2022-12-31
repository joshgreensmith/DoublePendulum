from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit

m = 1.0
l = 30.0
g = 9.81
Tmax = 100
Tnumber = Tmax + 1

theta1_val = 0.02
theta2_val = 1.224

@jit
def doublePend(y, t, l, m, g):
    # Define the differential equations
    theta1, theta2, p1, p2 = y

    dtheta1 = (6.0 * (2.0 * p1 - 3.0 * p2 * math.cos(theta1 - theta2))) / (m * l**2.0 * (16.0 - 9.0 * (math.cos(theta1 - theta2)**2.0)))
    dtheta2 = (6.0 * (2.0 * p2 - 3.0 * p1 * math.cos(theta1 - theta2))) / (m * l**2.0 * (16.0 - 9.0 * (math.cos(theta1 - theta2)**2.0)))

    dp1 = - 0.5 * m * l**2.0 * (dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3.0 * (g / l) * math.sin(theta1))
    dp2 = - 0.5 * m * l**2.0 * (-dtheta1 * dtheta2 * math.sin(theta1 - theta2) + 3.0 * (g / l) * math.sin(theta1))

    dydx = [dtheta1, dtheta2, dp1, dp2]
    return dydx


y0 = [theta1_val, theta2_val, 0.0, 0.0]
t = np.linspace(0, Tmax, Tnumber)
sol = odeint(doublePend, y0, t, args=(l,m,g), mxstep=5000000)

theta1 = []
theta2 = []
p1 = []
p2 = []

for i in sol:
    theta1.append(i[0])
    theta2.append(i[1])
    p1.append(i[2])
    p2.append(i[3])


# Create plots for both angles and phase space portrait for angles vs momenta of each pendula
fig, axs = plt.subplots(2,2)
axs[0, 0].plot(theta1)
axs[0, 0].set_title('Theta1 vs time')
axs[0, 0].set(xlabel='Time', ylabel='Theta 1')

axs[0, 1].plot(theta2, 'tab:orange')
axs[0, 1].set_title('Theta2 vs time')
axs[0, 1].set(xlabel='Time', ylabel='Theta 2')

axs[1, 0].plot(theta1, p1)
# axs[1, 0].plot(theta1, p1, 'o', color='black')
axs[1, 0].set_title('Theta1 vs p1 Phase Space')
axs[1, 0].set(xlabel='Theta1', ylabel='p1')
axs[1, 0].set_yticklabels([])

axs[1, 1].plot(theta2, p2, 'tab:orange')
# axs[1, 1].plot(theta2, p2, 'o', color='black')
axs[1, 1].set_title('Theta2 vs p2 Phase Space')
axs[1, 1].set(xlabel='Theta2', ylabel='p2')
axs[1, 1].set_yticklabels([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Theta1_' + str(round(theta1_val, 3)) + '_Theta2_' + str(round(theta2_val, 3)) + '_Tmax_' + str(Tmax))
# plt.show()

plot_title = 'Phase_space_theta1_' + str(round(theta1_val, 3)) + '_theta2_' + str(round(theta2_val, 3)) + '_Tmax_' + str(Tmax) + '.png'
plt.savefig(plot_title)


