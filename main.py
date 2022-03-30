from src.postscript import *
from functools import reduce
from random import choices
from math import floor, ceil, sqrt


dancer = "mbdtf5.jpeg"          # Dancer
pixels = "mbdtf4.jpeg"          # Pixels
sword = "mbdtf6.jpg"            # Sword


def main():

    def point_slope_y(_m, _x, _y):
        return lambda __y: (__y - _y) / _m + _x

    # Constants
    k = 10
    background = (255, 255, 255)

    pixelsd = read_image(dancer)
    pixelsp = read_image(pixels)
    pixelss = read_image(sword)

    # Width and height of original images
    p_width = len(pixelsd[0])
    p_height = len(pixelsd)

    width = sqrt(3) * k     # Full width of hexagon
    half = width / 2        # Half width
    k1 = k / 2              # Half the height of hexagon
    k2 = 3 * k1             # Commonly used constant, 1.5 height of hexagon

    r1 = floor((p_width - half) / width)
    r2 = floor((p_width - width) / width)
    h1 = floor((p_height - k) / k2)
    h = int(h1 * k2 + k1)

    if r1 == r2:
        w = int(r1 * width + half)
    else:
        w = int(max(r1, r2) * width)

    vertices = []
    flat_points = []
    y = k
    for i in range(h1):
        vertices.append([])
        if i & 1 == 0:
            r = r1
            x = half
        else:
            r = r2
            x = width
        for j in range(r):
            p = (x, y)
            vertices[i].append(p)
            flat_points.append(p)
            x += width
        y += k2

    n_points = len(flat_points)
    weights = [1] * n_points

    pixel_pairs = [(pixelsd, pixelsp), (pixelsp, pixelss), (pixelss, pixelsd)]

    for m in range(3):

        pixels1, pixels2 = pixel_pairs[m]

        p1 = choices(flat_points, weights=weights)[0]

        def weight_func(_p, _x, _y):
            return min(pow(_x - _p[0], 2) / 50000 + pow(_y - _p[1], 2) / 50000, 1)

        for i in range(len(weights)):
            weights[i] = min(weights[i], weight_func(p1, *flat_points[i]))

        p2 = choices(flat_points, weights=weights)[0]

        weight_sketch = PostscriptPy.stdcanvas(w, h, 255, 255, 255)
        weight_sketch.setstdfont()

        for p, weight in zip(flat_points, weights):
            weight_sketch.text(str(round(weight, 2)), *p)

        for t in range(0, 40):
            def f1(_x, _y):
                return t - (pow(_x - p1[0], 2) / 10000 + pow(_y - p1[1], 2) / 10000)

            def f2(_x, _y):
                return t - (pow(_x - p2[0], 2) / 10000 + pow(_y - p2[1], 2) / 10000)
            eps = PostscriptPy.stdcanvas(w, h, *background)
            for row in vertices:
                for x, y in row:

                    # Left side of hexagon, upper and lower slopes
                    upper_left_bound = point_slope_y(k1 / half, x, y + k)
                    lower_left_bound = point_slope_y(-k1 / half, x, y - k)

                    colors1 = (0, 0, 0)
                    colors2 = (0, 0, 0)
                    n_colors = 0

                    # Iterate the height of the hexagon top->down

                    # From top of hex to first vertex
                    for j in range(floor(y + k), ceil(y + k1), -1):

                        x3 = upper_left_bound(j)
                        for i in range(ceil(x3), floor(2 * x - x3)):
                            colors1 = tuple(a + b for a, b in zip(colors1, pixels1[j][i]))
                            colors2 = tuple(a + b for a, b in zip(colors2, pixels2[j][i]))
                            n_colors += 1

                    # From first vertex to second vertex, both sides are vertical so no slope
                    for j in range(floor(y + k1), ceil(y - k1), -1):
                        for i in range(ceil(x - half), floor(x + half)):
                            colors1 = tuple(a + b for a, b in zip(colors1, pixels1[j][i]))
                            colors2 = tuple(a + b for a, b in zip(colors2, pixels2[j][i]))
                            n_colors += 1

                    # From second vertex to bottom of hex
                    for j in range(floor(y - k1), ceil(y - k), -1):
                        x3 = lower_left_bound(j)
                        for i in range(ceil(x3), floor(2 * x - x3)):
                            colors1 = tuple(a + b for a, b in zip(colors1, pixels1[j][i]))
                            colors2 = tuple(a + b for a, b in zip(colors2, pixels2[j][i]))
                            n_colors += 1

                    # Average colors
                    colors1 = tuple(a / n_colors for a in colors1)
                    colors2 = tuple(a / n_colors for a in colors2)

                    v1 = max(min(1, max(f1(x, y), f2(x, y))), 0)

                    colors = tuple(a * (1 - v1) + b * v1 for a, b in zip(colors1, colors2))

                    v2 = 1 - sum(colors) / 765

                    y1 = v2 * k2 / 2
                    x1 = v2 * half / 2
                    x2 = v2 * half

                    star = [
                        (x, h - (y + k)),
                        (x + x1, h - (y + y1)),
                        (x + half, h - (y + k1)),
                        (x + x2, h - y),
                        (x + half, h - (y - k1)),
                        (x + x1, h - (y - y1)),
                        (x, h - (y - k)),
                        (x - x1, h - (y - y1)),
                        (x - half, h - (y - k1)),
                        (x - x2, h - y),
                        (x - half, h - (y + k1)),
                        (x - x1, h - (y + y1))
                    ]
                    eps.fill_poly(star, *colors)

            eps.draw()
            return

        for i in range(len(weights)):
            weights[i] = min(weights[i], weight_func(p2, *flat_points[i]))

        for p, weight in zip(flat_points, weights):
            weight_sketch.text(str(round(weight, 2)), *p)

        weight_sketch.out(f"weight_sketch{m}.eps")


if __name__ == "__main__":
    main()
    # options = ["-delay", "20", "-density", "200"]
    # make_gif_magick(pattern="pspy.*[.]eps", options=options, sort_key=lambda s: int(s.split('.')[0][4:]))
