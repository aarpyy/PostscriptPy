from src.postscript import *
from functools import reduce
from random import choices


dancer = "mbdtf5.png"           # Dancer
pixels = "mbdtf4.jpeg"          # Pixels
sword = "mbdtf6.jpg"            # Sword


def main():

    # Constants
    k = 10
    background = (255, 255, 255)
    # t = 10

    pixelsd = read_image(dancer)
    pixelsp = read_image(pixels)
    pixelss = read_image(sword)

    width = sqrt(3) * k  # Full width of hexagon
    half = width / 2  # Half width
    k1 = k / 2  # Half the height of hexagon

    vertices = [[], []]  # First 2 rows empty

    x = half  # Center of first hexagon is at (half, k)

    # Width and height of original images
    p_width = len(pixelsd[0])
    p_height = len(pixelsd)

    # While the right edge of the current hex is within bounds of image, increment
    # x by the width of hexagon to find the number of whole hexagons that fit
    while x + half <= p_width:
        vertices[0].append((x, k))
        x += width

    w = x - half

    x = width
    y = 5 * k1
    while x + half <= p_width:
        vertices[1].append((x, y))
        x += width

    # Width value is set so that hexagons fit neatly
    w = ceil(max(w, x - half))

    i = 0
    y = 5 * k1
    while 1:
        y += 3 * k1
        if y + k >= p_height:
            break

        vertices.append([(x, y) for x, _ in vertices[i]])
        i += 1

    # Height set also so that hexagons fit neatly
    h = ceil(y - k1)
    flat_points = reduce(lambda r, c: r + c, vertices)
    weights = [1] * len(flat_points)

    def point_slope_y(_m, _x, _y):
        return lambda __y: (__y - _y) / _m + _x

    pixel_pairs = [(pixelsd, pixelsp), (pixelsp, pixelss), (pixelss, pixelsd)]

    for m in range(3):

        pixels1, pixels2 = pixel_pairs[m]

        p1 = choices(flat_points, weights=weights)[0]

        def weight_func(_x, _y, _p):
            return min(pow(_x - _p[0], 2) / 50000 + pow(_y - _p[1], 2) / 50000, 1)

        for i in range(len(weights)):
            weights[i] = min(weights[i], weight_func(*flat_points[i], p1))

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

                    dx = half
                    dy = k1

                    # Left side of hexagon, upper and lower slopes
                    upper_left_bound = point_slope_y(dy / dx, x, y + k)
                    lower_left_bound = point_slope_y(-dy / dx, x, y - k)

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

                    y1 = v2 * 3 * k / 4
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

            eps.out()

        for i in range(len(weights)):
            weights[i] = min(weights[i], weight_func(*flat_points[i], p2))

        for p, weight in zip(flat_points, weights):
            weight_sketch.text(str(round(weight, 2)), *p)

        weight_sketch.out(f"weight_sketch{m}.eps")


if __name__ == "__main__":
    # main()
    options = ["-delay", "20", "-density", "200"]
    make_gif_magick(pattern="pspy.*[.]eps", options=options, sort_key=lambda s: int(s.split('.')[0][4:]))
