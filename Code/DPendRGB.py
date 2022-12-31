from PIL import Image
import math
import time
import numpy as np
from scipy.integrate import odeint
from numba import jit


gridSize = 150
start_time = time.time()

img = Image.new('RGB', (gridSize, gridSize))

m = 1
l = 100
g = 9.81

Tmax = 100
Tnumber = 101

# Automate whole process + output a lot of graphs on raspberry pi over time
# Comvert colour system from rgb to hsv using colour conversion
# https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion


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
    sol = odeint(doublePend, y0, t, args=(l, m, g))

    # Extract theta2 from the set of solutions
    theta2Solution = []
    for i in sol:
        theta2Solution.append(i[1])

    # Test whether theta2 has crossed the boundary conditions and flipped in the time, check better
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
    return 0



def colourConversion(flipTime):
    if flipTime < 10 * (l / g)**0.5:
        return [0, 255, 0]
    if flipTime < 100 * (l / g)**0.5:
        return [255, 0, 0]
    if flipTime < 1000 * (l / g)**0.5:
        return [255, 255, 0]
    if flipTime < 10000 * (l / g)**0.5:
        return [0, 0, 255]
    return [255, 255, 255]


for x in range(gridSize):
    print("x = " + str(x))
    for y in range(gridSize):
        angle1 = ((gridSize / 2) - x) * (3 / (gridSize / 2))
        angle2 = ((gridSize / 2) - y) * (3 / (gridSize / 2))

        flipTime = doublePendTest(angle1, angle2)
        # print(flipTime)
        # colourArray = colourConversion(flipTime)

        # img.putpixel((x, y), (colourArray[0], colourArray[1], colourArray[2]))
        pixelColour = int(flipTime * 255 / 50)
        img.putpixel((x, y), (pixelColour, 50, 50))

name = "length" + str(l)  + "_mass" + str(m) + "_res" + str(gridSize) + ".png"
img.save(name)
print("--- %s seconds ---" % (time.time() - start_time))
