from PIL import Image
import math
import time
import numpy
import scipy


gridSize = 50
start_time = time.time()


img = Image.new('RGB', (gridSize, gridSize))

g = 9.81
l1 = 100
l2 = 100
m1 = 10
m2 = 10
timeFactor = 0.15


def runExperiment(Theta1, Theta2, l1, l2, m1, m2, timeFactor):
    dTheta1 = 0
    dTheta2 = 0
    d2Theta1 = 0
    d2Theta2 = 0

    for timeSteps in range(int(1000 * (l1 / g)**0.5)):
        prevTheta2 = Theta2
        mu = 1 + (m1 / m2)
        d2Theta1 = (g * (math.sin(Theta2) * math.cos(Theta1 - Theta2) - mu * math.sin(Theta1)) - (l2 * dTheta2 * dTheta2 + l1 * dTheta1 *
                                                                                                  dTheta1 * math.cos(Theta1 - Theta2)) * math.sin(Theta1 - Theta2)) / (l1 * (mu - math.cos(Theta1 - Theta2) * math.cos(Theta1 - Theta2)))
        d2Theta2 = (mu * g * (math.sin(Theta1) * math.cos(Theta1 - Theta2) - math.sin(Theta2)) + (mu
                                                                                                  * l1 * dTheta1 * dTheta1 + l2 * dTheta2 *
                                                                                                  dTheta2 * math.cos(Theta1 - Theta2)) * math.sin(Theta1 - Theta2)) / (l2 * (mu - math.cos(Theta1 - Theta2) * math.cos(Theta1 - Theta2)))
        dTheta1 += d2Theta1 * timeFactor
        dTheta2 += d2Theta2 * timeFactor
        Theta1 += dTheta1 * timeFactor
        Theta2 += dTheta2 * timeFactor

        prevRotatedPendulum2Bearing = (prevTheta2 * 360 / (2 * math.pi)) % 360
        rotatedPendulum2Bearing = (Theta2 * 360 / (2 * math.pi)) % 360

        if prevRotatedPendulum2Bearing < 180 and prevRotatedPendulum2Bearing > 150 and rotatedPendulum2Bearing > 180:
            return timeSteps * timeFactor

        if prevRotatedPendulum2Bearing > 180 and prevRotatedPendulum2Bearing < 210 and rotatedPendulum2Bearing < 180:
            return timeSteps * timeFactor

    return -1


def colourConversion(flipTime):
    if flipTime < 10 * (l1 / g)**0.5:
        return [0, 255, 0]
    if flipTime < 100 * (l1 / g)**0.5:
        return [255, 0, 0]
    if flipTime < 1000 * (l1 / g)**0.5:
        return [255, 255, 0]
    if flipTime < 10000 * (l1 / g)**0.5:
        return [0, 0, 255]
    return [255, 255, 255]


for x in range(gridSize):
    print("x = " + str(x))
    for y in range(gridSize):
        angle1 = ((gridSize / 2) - x) * (3 / (gridSize / 2))
        angle2 = ((gridSize / 2) - y) * (3 / (gridSize / 2))

        flipTime = runExperiment(angle1, angle2, l1, l2, m1, m2, timeFactor)
        # colourArray = colourConversion(flipTime)

        # img.putpixel((x, y), (colourArray[0], colourArray[1], colourArray[2]))
        img.putpixel(
            (x, y), (int(flipTime * 255 / (int(10 * (l1 / g)**0.5))), 50, 50))

img.save("pendulumLoop.png")
print("--- %s seconds ---" % (time.time() - start_time))
