from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
import pygame

m = 1.0
l = 30.0
g = 9.81
Tmax = 100
Tnumber = Tmax + 1

theta1_val = 3.2
theta2_val = 0.1

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

print(theta1)

# START PYGAME ANIMATION CODE

pygame.init()
screen = pygame.display.set_mode((1000, 1000))
done = False

clock = pygame.time.Clock()

angle = 0
length = 100
counter = 0
index = 0

def draw_line_at_angle(angle):
    x = 500 + math.cos(angle) * length
    y = 500 + math.sin(angle) * length

    pygame.draw.line(screen, (255,255,255), (500,500), (x,y), 1)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill((0,0,0))

    counter += 60

    if counter > 300:
        index += 1
        counter = 0

    print(index)

    angle1 = theta1[index]
    angle2 = theta2[index]

    x1 = 500 + math.cos(angle1) * length
    y1 = 500 + math.sin(angle1) * length

    x2 = x1 + math.cos(angle2) * length
    y2 = y1 + math.sin(angle2) * length

    print(angle1, angle2)
    print(x1, y1, x2, y2)

    pygame.draw.line(screen, (255,255,255), (500,500), (x1,y1), 1)
    pygame.draw.line(screen, (255,255,255), (x1,y1), (x2,y2), 1)
    
    pygame.display.flip()
    clock.tick(60)
