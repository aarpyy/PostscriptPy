from pathlib import Path
from os import system
from PIL import Image
from random import randrange
from math import sin, cos, radians, atan, degrees, tan, sqrt, floor, ceil
from typing import Sequence, Callable
from collections import deque


def read_image(file) -> list[list[tuple[int, int, int]]]:
    """
    Returns a list of lines where each line contains
    a tuple (r, g, b) for each pixel in the corresponding row of the input image.

    :param file: image
    :return: string with pixel data
    """

    # If relative path, check to see if it's in the images folder
    if "/" not in file:
        images = PostscriptPy.root.joinpath("images")
        if not images.joinpath(file).is_file():
            raise ValueError("Relative path to image not found")
        else:
            file = str(images.joinpath(file))

    img = Image.open(file)  # type: Image.Image

    buffer = []

    # Remove alpha value since postscript doesn't have alphas
    data = list(pixel[:-1] for pixel in img.getdata())
    w = img.width
    for i in range(img.height):
        buffer.append(data[i * w:(i + 1) * w])
    return buffer


def write_img_data(file):
    """
    Extension of img_to_data() that writes the output list to a file in the
    pixels folder of the project.

    :param file: image
    :return: None
    """
    path = str(PostscriptPy.root.joinpath("pixels"))
    with open(f"{path}.pixel", "w") as out:
        out.writelines('|'.join(line) for line in read_image(file))


class PostscriptPy:
    RECT = 1
    RECT_FILL = 2
    SQUARE = 3
    SQUARE_FILL = 4

    watermark = "%Made by Postcript.py"

    root = Path(__file__).parent.parent

    # Path for save files
    path_out = root.joinpath("out")

    # Path for a temp file, used for displaying image once, files in temp will be overwritten
    path_temp = root.joinpath("temp")

    def __init__(self, *args, _load: bool = False, _buffer: str = None, _ref: list[list[tuple[int, int, int]]] = None):
        """
        Constructs new instance of PostscriptPy object based on image size or
        a pre-constructed buffer.

        :param args: image size (w, h) for rect image, (w) for square
        :param _load: if constructor is being used to load from pre-written buffer
        :param _buffer: pre-written buffer
        """
        if _load:

            # Don't include %EOF or showpage since those should be last and prevent new stuff from being added easily
            lines = [
                line for line in _buffer.split() if not line.startswith("%EOF") and not line.startswith("showpage")
            ]
            for line in lines:
                if line.startswith("%%Bounding Box: "):
                    size = line.split(" ")
                    self.w = int(size[3])
                    self.h = int(size[4])

            self._buffer = "\n".join(lines)
        else:
            if _ref is None:
                match len(args):
                    case 1:
                        self.w = self.h = args[0]
                    case 2:
                        self.w = args[0]
                        self.h = args[1]
            else:
                self.w = len(_ref[0])
                self.h = len(_ref)

            self._buffer = f"{PostscriptPy.watermark}\n" \
                           f"%!PS-Adobe-3.0 EPSF-3.0\n" \
                           f"%%BoundingBox: 0 0 {self.w} {self.h}\n"

        self.defined = set()
        self._color = None
        self._ref = _ref

    def __str__(self):
        return self._buffer

    @property
    def width(self):
        return self.w

    @property
    def height(self):
        return self.h

    @staticmethod
    def from_image(fp: str):
        return PostscriptPy(_ref=read_image(fp))

    def fill_squares(
            self,
            *,
            k: int = 20,
            gray: bool = False,
            neg: bool = False,
            rndm: bool = False,
            size: int = None,
            saddle: Callable = None,
    ):
        """
        Creates a PostscriptPy object based on an input image by printing squares with color
        as the average of the pixels in the kxk area surrounding. The size of the square
        is determined by the average brightness of the color.

        :param k: square size
        :param gray: grayscale
        :param neg: size is negated so that darker color is larger square
        :param rndm: random size of square
        :param size: (with rndm) maximum size a random square can be
        :param saddle: use a saddle function to determine size of squares
        :return: PostscriptPy object with squares drawn
        """

        if self._ref is None:
            raise ValueError(f"Instances needs to be created from a reference image "
                             f"using {PostscriptPy.from_image.__name__}")
        elif rndm and size is None:
            raise ValueError("Random images need specified size range")

        use_saddle = saddle is not None

        # This ensures that there isn't blank canvas if k does not divide w and h
        w = (len(self._ref[0]) // k) * k
        h = (len(self._ref) // k) * k

        k_sq = k * k
        k_2 = k // 2

        cols = w // k
        rows = h // k

        for i in range(cols):
            ik = i * k
            for j in range(rows):
                jk = j * k
                colors = (0, 0, 0)

                # Iterate over the range of the block size
                for ip in range(k):
                    for jp in range(k):
                        # Accumulate the (r, g, b) values, pixels is list of rows so give row number first
                        colors = tuple(x + y for x, y in zip(colors, self._ref[jk + jp][ik + ip]))

                # Take the average of the pixel value (k * k total pixels)
                colors = tuple(x // k_sq for x in colors)

                if use_saddle:
                    size = saddle(i, j)
                elif rndm:
                    # Assign random size that is from k-size to k
                    size = randrange(1, k)
                elif gray:
                    # Set color to be black and size to correlate to brightness value
                    avg = sum(colors) / 3
                    colors = (0, 0, 0)
                    size = round(k * ((255 - avg) / 255), 3)
                elif neg:
                    # If negate, correlate larger with lighter colors
                    size = round((sum(colors) / (3 * 255)) * k, 3)
                else:
                    # Otherwise darker colors means larger
                    size = round(((255 - (sum(colors) / 3)) / 255) * k, 3)

                self.fill_square(*PostscriptPy.center_square(ik + k_2, h - (jk + k_2), size), *colors)

    @classmethod
    def from_image_flexible(
            cls,
            file: str,
            *,
            k: int = 20,
            bckgrnd: tuple[int, int, int] = None,
            gray: bool = False,
            neg: bool = False,
            rndm: bool = False,
    ):

        point_slope_y = lambda m, _x, _y: lambda __y: (__y - _y) / m + _x

        pixels = read_image(file)

        width = sqrt(3) * k
        half = width / 2
        k1 = k / 2

        # First 2 rows empty
        vertices = [[], []]

        x = half

        # While the right edge of the current hex is within bounds of image
        while x + half < len(pixels[0]):
            vertices[0].append((x, k))
            x += width

        x = width
        y = 5 * k1
        while x + half < len(pixels[0]):
            vertices[1].append((x, y))
            x += width

        i = 0
        y = 5 * k1
        while 1:
            y += 3 * k1
            if y + k >= len(pixels):
                break

            vertices.append([(x, y) for x, _ in vertices[i]])
            y += 3 * k1
            if y + k >= len(pixels):
                break

            i += 1
            vertices.append([(x, y) for x, _ in vertices[i]])
            i += 1

        h = k * (3 * len(vertices) + 1) / 2
        w = width * (len(vertices[0]) + 0.5)

        eps = cls(w, h)
        if bckgrnd is not None:
            eps.color(*bckgrnd)
            eps.fill_rect(0, 0, w, h)
        eps.setlinejoin()
        eps.setlinecap()
        eps.setlinewidth(2)

        for row in vertices:
            for x, y in row:
                dx = half
                dy = k1

                upper_left_bound = point_slope_y(dy / dx, x, y + k)
                lower_left_bound = point_slope_y(-dy / dx, x, y - k)

                colors = (0, 0, 0)
                n_colors = 0

                # Iterate the height of the hexagon top->down
                for j in range(floor(y + k), ceil(y + k1), -1):

                    x3 = upper_left_bound(j)
                    for i in range(ceil(x3), floor(2 * x - x3)):
                        colors = tuple(a + b for a, b in zip(colors, pixels[j][i]))
                        n_colors += 1

                for j in range(floor(y + k1), ceil(y - k1), -1):
                    for i in range(ceil(x - half), floor(x + half)):
                        colors = tuple(a + b for a, b in zip(colors, pixels[j][i]))
                        n_colors += 1

                for j in range(floor(y - k1), ceil(y - k), -1):
                    x3 = lower_left_bound(j)
                    for i in range(ceil(x3), floor(2 * x - x3)):
                        colors = tuple(a + b for a, b in zip(colors, pixels[j][i]))
                        n_colors += 1

                colors = tuple(a / n_colors for a in colors)

                # sum(colors) / 3 / 255
                v = sum(colors) / 765
                y1 = v * 3 * k / 4
                x1 = v * half / 2
                x2 = v * half

                # x, y = point
                # eps.draw_line(x, y, x2=x, y2=y)
                # hexagon = [(x, y + k), (x + width / 2, y + k / 2), (x + width / 2, y - k / 2),
                #            (x, y - k), (x - width / 2, y - k / 2), (x - width / 2, y + k / 2)]
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

        return eps

    def curve_fractal(self, depth: int = 2, f_h: float = 0.5, f_l: float = 0.5):

        from math import pi

        _2pi = 2 * pi

        polys = deque()
        polys.appendleft(((0, 0), (self.w, self.h * f_h), (self.w, 0)))

        point_slope_x = lambda m, x1, y1: lambda x: m * (x - x1) + y1
        point_slope_y = lambda m, x1, y1: lambda y: (y - y1) / m + x1

        dy = self.h * f_h
        d_theta = atan(dy / self.w)

        theta = d_theta

        # Point that next line must cross through
        point = (self.w * f_l, dy * f_l)

        for i in range(1, depth):
            # Rotate by increment of theta
            theta = (theta + d_theta) % _2pi

            f_x = point_slope_x(tan(theta), *point)
            f_y = point_slope_y(tan(theta), *point)

            p1, p2 = self.get_intercepts(f_x, f_y)
            point = ((p1[0] + p2[0]) * f_l, (p1[1] + p2[1]) * f_l)

            polys.appendleft(self.get_bounding(p1, p2, theta))

        gray = 0
        d_gray = min(255 / depth, 5)
        for poly in polys:
            self.fill_poly(poly, gray)
            gray += d_gray

    def curve_fractal_square(self, depth=20, f_h=.07, f_l=.5):
        self.curve_fractal(depth=depth, f_h=f_h, f_l=f_l)

        self.translate(self.w, 0)
        self.rotate(90)
        self.curve_fractal(depth=depth, f_h=f_h, f_l=f_l)

        self.translate(self.w, self.h)
        self.rotate(180)
        self.curve_fractal(depth=depth, f_h=f_h, f_l=f_l)

        self.translate(0, self.h)
        self.rotate(270)
        self.curve_fractal(depth=depth, f_h=f_h, f_l=f_l)

    def get_intercepts(self, fx: Callable, fy: Callable, U=None, D=0, L=0, R=None):
        if U is None:
            U = self.h
        if R is None:
            R = self.w

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

    def get_bounding(self, p1, p2, theta, U=None, D=0, L=0, R=None):

        if U is None:
            U = self.h
        if R is None:
            R = self.w

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
        if p1[1] == D:
            i1 = 4
        elif p1[0] == L:
            i1 = 1
        elif p1[1] == U:
            i1 = 2
        else:
            i1 = 3

        # Find where p2 fits in, in cyclic order
        if p2[1] == D:
            i2 = 4
        elif p2[0] == L:
            i2 = 1
        elif p2[1] == U:
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

    def out(self, file=None):

        if file is None:
            n = 0
            for p in self.path_out.iterdir():
                if p.is_file() and str(p).split('/')[-1].startswith("pspy"):
                    n = max(n, int(str(p).split('/')[-1].split(".")[0][4:]))
            file = str(self.path_out.joinpath(f"pspy{n + 1}.eps"))

        self._buffer += "showpage\n%EOF"
        with open(file, "w") as psfile:
            psfile.write(self._buffer)

    def draw(self):
        file = str(self.path_temp.joinpath(f"image.eps"))
        self.out(file)
        system(f"open -a Preview {file}")

    def translate(self, x, y):
        self._buffer += f"{x} {y} translate\n"

    def setlinejoin(self, value=1):
        self._buffer += f"{value} setlinejoin\n"

    def setlinecap(self, value=1):
        self._buffer += f"{value} setlinecap\n"

    def setlinewidth(self, value):
        self._buffer += f"{value} setlinewidth\n"

    def color(self, r, g=None, b=None):
        if g is None or r == g == b:
            gray = round(r / 255, 3)
            if gray != self._color:
                self._buffer += f"{gray} setgray\n"
                self._color = gray
        else:
            r = round(r / 255, 3)
            g = round(g / 255, 3)
            b = round(b / 255, 3)
            if (r, g, b) != self._color:
                self._buffer += f"{r} {g} {b} setrgbcolor\n"
                self._color = (r, g, b)

    def newpath(self, *args):
        self._buffer += "newpath\n"

        if args:
            if len(args) == 2:
                x = str(args[0])
                y = str(args[1])
                self._buffer += (" " * (4 - len(x))) + x + (" " * (4 - len(y))) + y + " moveto\n"
            else:
                raise ValueError(f"Invalid {PostscriptPy.newpath.__name__} args: {args}")

    def rotate(self, theta):
        self._buffer += f"{theta} rotate\n"

    def draw_line(
            self, x: int | float, y: int | float, r: int = None, g: int = None, b: int = None,
            *, x2: int | float = None, y2: int | float = None, length: int = None, theta: int | float = None
    ):
        if isinstance(length, int) and isinstance(theta, (int, float)):
            x2 = x + int(length * cos(radians(theta)))
            y2 = y + int(length * sin(radians(theta)))
        if isinstance(x2, int | float) and isinstance(y2, int | float):
            if isinstance(r, int | float):
                self.color(r, g, b)
            self._buffer += f"newpath\n" \
                            f"{x} {y} moveto\n" \
                            f"{x2} {y2} lineto\n" \
                            f"closepath\n" \
                            f"stroke\n"
        else:
            raise ValueError("Must provide either end desination or length and angle")

    def draw_rect(self, x: int | float, y: int | float, w: int | float, h: int | float,
                  r: int = None, g: int = None, b: int = None):
        if PostscriptPy.RECT not in self.defined:
            self._buffer += (
                    "/draw_rect {\n" +
                    "/h exch def\n" +  # Exchange top two on stack after adding /h results in /h val def
                    "/w exch def\n" +
                    "/y exch def\n" +
                    "/x exch def\n" +
                    "newpath\n" +
                    "x y moveto\n" +
                    "x w add y lineto\n" +
                    "x w add y h add lineto\n" +
                    "x y h add lineto\n" +
                    "closepath\n" +
                    "stroke\n" +
                    "} def\n"
            )
            self.defined.add(PostscriptPy.RECT)

        if isinstance(r, int | float):
            self.color(r, g, b)
        self._buffer += f"{x} {y} {w} {h} draw_rect\n"

    def fill_rect(self, x: int | float, y: int | float, w: int | float, h: int | float,
                  r: int = None, g: int = None, b: int = None):
        if PostscriptPy.RECT_FILL not in self.defined:
            self._buffer += (
                    "/fill_rect {\n" +
                    "/h exch def\n" +  # Exchange top two on stack after adding /h results in /h val def
                    "/w exch def\n" +
                    "/y exch def\n" +
                    "/x exch def\n" +
                    "newpath\n" +
                    "x y moveto\n" +
                    "x w add y lineto\n" +
                    "x w add y h add lineto\n" +
                    "x y h add lineto\n" +
                    "closepath\n" +
                    "fill\n" +
                    "} def\n"
            )
            self.defined.add(PostscriptPy.RECT_FILL)

        if isinstance(r, int | float):
            self.color(r, g, b)
        self._buffer += f"{x} {y} {w} {h} fill_rect\n"

    def draw_square(self, x: int | float, y: int | float, w: int | float,
                    r: int = None, g: int = None, b: int = None):
        if PostscriptPy.SQUARE not in self.defined:
            self._buffer += (
                    "/draw_square {\n" +
                    "/w exch def\n" +  # Exchange top two on stack after adding /w results in /w val def
                    "/y exch def\n" +
                    "/x exch def\n" +
                    "newpath\n" +
                    "x y moveto\n" +  # Start bottom left, move counter clockwise
                    "x w add y lineto\n" +
                    "x w add y w add lineto\n" +
                    "x y w add lineto\n" +
                    "closepath\n" +
                    "stroke\n" +
                    "} def\n"
            )
            self.defined.add(PostscriptPy.SQUARE)

        if isinstance(r, int | float):
            self.color(r, g, b)
        self._buffer += f"{x} {y} {w} draw_square\n"

    def fill_square(self, x: int | float, y: int | float, w: int | float,
                    r: int = None, g: int = None, b: int = None):
        if PostscriptPy.SQUARE_FILL not in self.defined:
            self._buffer += (
                    "/fill_square {\n" +
                    "/w exch def\n" +  # Exchange top two on stack after adding /w results in /w val def
                    "/y exch def\n" +
                    "/x exch def\n" +
                    "newpath\n" +
                    "x y moveto\n" +
                    "x w add y lineto\n" +
                    "x w add y w add lineto\n" +
                    "x y w add lineto\n" +
                    "closepath\n" +
                    "fill\n" +
                    "} def\n"
            )
            self.defined.add(PostscriptPy.SQUARE_FILL)

        if isinstance(r, int | float):
            self.color(r, g, b)
        self._buffer += f"{x} {y} {w} fill_square\n"

    def __poly(self, vertices: Sequence, r: int = None, g: int = None, b: int = None):
        if isinstance(r, int | float):
            self.color(r, g, b)
        self._buffer += (
                f"newpath\n" +
                f"{vertices[0][0]} {vertices[0][1]} moveto\n"
        )
        for i in range(1, len(vertices)):
            self._buffer += f"{vertices[i][0]} {vertices[i][1]} lineto\n"
        self._buffer += "closepath\n"

    def draw_poly(self, vertices: Sequence, r: int = None, g: int = None, b: int = None):
        self.__poly(vertices, r, g, b)
        self._buffer += "stroke\n"

    def fill_poly(self, vertices: Sequence, r: int = None, g: int = None, b: int = None):
        self.__poly(vertices, r, g, b)
        self._buffer += "fill\n"

    @staticmethod
    def center_rect(x: int, y: int, w: int, h: int) -> tuple[int | float, int | float, int, int]:
        """Helper function to give top left coordinates of rectangle with center at (x, y)"""
        return x - (w / 2), y - (h / 2), w, h

    @staticmethod
    def center_square(x: int, y: int, w: int) -> tuple[int | float, int | float, int]:
        """Helper function to give top left coordinates of square with center at (x, y)"""
        return x - (w / 2), y - (w / 2), w


def load(file: str) -> PostscriptPy:
    ftype = file.split(".")
    if ftype != "eps":
        raise ValueError(f"{file} is not a valid postscript file!")
    else:
        with open(file, "r") as f:
            for line in f:
                if line:
                    if not line.startswith(PostscriptPy.watermark):
                        raise ValueError(f"Not a valid {PostscriptPy.__class__.__name__} file!")
                    else:
                        return PostscriptPy(_load=True, _buffer=f.read())
