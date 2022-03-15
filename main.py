from postscript import *
from math import gcd

w = h = 100

# Bounds for box, left, right, down, up (inclusive)
U = h
D = 0
L = 0
R = w


def get_intercepts(fx: Callable, fy: Callable):
    # If line intersects the lower bound within the right and left bounds
    if R >= fy(D) >= L:
        p1 = (fy(D), D)

        # Find second intersection among upper, right, and left bounds
        if R >= fy(U) >= L:
            p2 = (fy(U), U)
        elif U >= fx(R) >= D:
            p2 = (R, fx(R))
        else:
            p2 = (L, fx(L))

    # If line intersects left bound between upper and lower
    elif U >= fx(L) >= D:
        p1 = (L, fx(L))

        # Find second intersection among upper and right bounds since it wasn't down
        if R >= fy(U) >= L:
            p2 = (fy(U), U)
        else:
            p2 = (R, fx(R))

    # Since the line MUST intersect two sides, if it didn't intersect either left or lower, it must be upper and right
    else:
        p1 = (R, fx(R))
        p2 = (fy(U), U)

    return p1, p2


def get_bounding(p1, p2, theta):
    from math import pi

    # If angle is between 0° and 135°, then the largest y value is considered the 'first' point
    if theta < 3 * pi / 4:
        if p1[1] < p2[1]:
            p1, p2 = p2[:], p1[:]

    # If angle is between 135° and 225°, then 'first' is lowest x
    elif theta < 5 * pi / 4:
        if p1[0] > p2[0]:
            p1, p2 = p2[:], p1[:]

    # If angle between 225° and 305°, then 'first' is lowest y
    elif theta < 7 * pi / 4:
        if p1[1] > p2[1]:
            p1, p2 = p2[:], p1[:]

    # Otherwise, if angle between 305° and 360°, then 'first' is largest x
    else:
        if p1[0] < p2[0]:
            p1, p2 = p2[:], p1[:]

    # Vertices of bounding box
    vertices = [(L, D), (L, U), (R, U), (R, D)]

    # Find where p1 fits in, in cyclic order
    if p1[1] == 0:
        i1 = 4
    elif p1[0] == 0:
        i1 = 1
    elif p1[1] == h:
        i1 = 2
    else:
        i1 = 3

    # Find where p2 fits in, in cyclic order
    if p2[1] == 0:
        i2 = 4
    elif p2[0] == 0:
        i2 = 1
    elif p2[1] == h:
        i2 = 2
    else:
        i2 = 3

    if i2 < i1:
        vertices.insert(i1, p1)
        vertices.insert(i2, p2)
        i1 += 1
    else:
        vertices.insert(i2, p2)
        vertices.insert(i1, p1)
        i2 += 1

    polys = []

    i = i1
    while i != i2:
        polys.append(vertices[i])
        i = (i + 1) % 6

    polys.append(vertices[i2])
    return polys


if __name__ == "__main__":
    # eps = PostscriptPy(300, 300)
    #
    # eps.setlinejoin()
    # eps.setlinecap()
    # eps.setlinewidth(2)
    # eps.draw_line(50, 50, x2=50, y2=50)
    eps = PostscriptPy.from_image_flexible("mbdtf.png", bckgrnd=(0, 0, 0))
    eps.draw()

