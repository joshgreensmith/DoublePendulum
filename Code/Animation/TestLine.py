import pygame
import math

pygame.init()
screen = pygame.display.set_mode((1000, 1000))
done = False

clock = pygame.time.Clock()

angle = 0
length = 100


def draw_line_at_angle(angle):
    x = 500 + math.cos(angle) * length
    y = 500 + math.sin(angle) * length

    pygame.draw.line(screen, (255,255,255), (500,500), (x,y), 1)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill((0,0,0))

    angle = (angle + pygame.time.get_ticks() * 0.00001)
    print(angle)
    draw_line_at_angle(angle)


        
    pygame.display.flip()
    clock.tick(60)
