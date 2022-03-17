import subprocess
import re
import platform
import sys

from enum import Enum
from pathlib import Path
from PIL import Image
from random import randrange, uniform
from math import sin, cos, radians, atan, tan, sqrt, floor, ceil
from typing import Sequence, Callable
from collections import deque
from perlin_noise import PerlinNoise
from numpy import shape


def read_image(fp: str) -> list[list[tuple[int, int, int]]]:
    """
    Returns a list of lines where each line contains
    a tuple (r, g, b) for each pixel in the corresponding row of the input image.

    :param fp: image
    :return: string with pixel data
    """

    file = Path(fp)

    # If relative path, check to see if it's in the images folder
    if not file.is_absolute():
        images = PostscriptPy.source.joinpath("images")
        if not images.joinpath(file).is_file():
            raise ValueError("Relative path to image not found")
        else:
            file = images.joinpath(fp)

    img = Image.open(str(file))  # type: Image.Image

    buffer = []

    if fp.endswith(".png"):
        # Remove alpha value since postscript doesn't have alphas
        data = list(pixel[:-1] for pixel in img.getdata())
    else:
        data = list(img.getdata())

    w = img.width
    for i in range(img.height):
        buffer.append(data[i * w:(i + 1) * w])
    return buffer


def make_gif(fp: str, save: str = None, pattern: str = None):
    path = Path(fp)
    if not path.is_absolute():
        path = PostscriptPy.source.joinpath(path)

    if pattern is None:
        files = sorted(list(path.iterdir()))
    else:
        r = re.compile(pattern)
        files = sorted(list(f for f in path.iterdir() if r.match(f.name)))

    frames = []
    for file in files:
        frames.append(Image.open(file))
    if save is None:
        n = 0
        for f in Path(".").iterdir():
            if f.name.startswith("gif"):
                n = max(n, int(f.name.split(".")[0][3:]))
        save = f"gif{n + 1}.gif"

    frames[0].save(save, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=10 * len(frames), loop=0)


class PostscriptPy(object):
    class EPS(Enum):
        RECT = 1
        RECT_FILL = 2
        SQUARE = 3
        SQUARE_FILL = 4

    CURRENT_VERSION = 1.1

    WATERMARK = "%Made by Postcript.py"
    __WATERMARK_1_1 = WATERMARK + " v1.1"
    __WATERMARK_1_2 = WATERMARK + " v1.2"

    __WATERMARKS = {
        1.0: WATERMARK,
        1.1: __WATERMARK_1_1,
        1.2: __WATERMARK_1_2
    }

    __BGST = "%BACKGROUNDSTART"
    __BGEND = "%BACKGROUNDEND"
    __FRST = "%FRAMESTART"
    __FREND = "%FRAMEEND"

    source = Path(__file__).parent.parent

    # Path for save files
    __path_out = source.joinpath("out")

    # Path for a temp file, used for displaying image once, files in temp will be overwritten
    __path_temp = source.joinpath("temp")

    def __init__(
            self,
            *args,
            _load: bool = False,
            _buffer: str = None,
            _ref: list[list[tuple[int, int, int]]] = None
    ):
        """
        Constructs new instance of PostscriptPy object based on image size or
        a pre-constructed buffer.

        :param args: image size (w, h) for rect image, (w) for square
        :param _load: if constructor is being used to load from pre-written buffer
        :param _buffer: pre-written buffer
        :param _ref: reference image, given in 2D list of pixel values
        """

        if _load:
            lines = _buffer.split("\n")
            if "v" in lines[0]:
                self.__version__ = float(lines[0].split("v")[1])
            else:
                self.__version__ = 1.0
            for line in lines:
                if line.startswith("%%BoundingBox:"):
                    size = line.split(" ")
                    self.w = int(size[3])
                    self.h = int(size[4])

            # Remove and keep track of frame separately
            if self.__version__ >= 1.2:
                start = lines.index(PostscriptPy.__FRST)
                end = lines.index(PostscriptPy.__FREND)
                self._frame = "\n".join(lines[start + 1:end]) + "\n"
                lines = lines[:start - 1] + lines[end:]
            else:
                self._frame = ""

            # Don't include %EOF or showpage since those should be last and prevent new stuff from being added easily
            self._buffer = "\n".join(lines[:-2]) + "\n"
        else:
            if _ref is None:
                match len(args):
                    case 1:
                        self.w = self.h = args[0]
                    case 2:
                        self.w = args[0]
                        self.h = args[1]
                    case _:
                        raise ValueError(f"Invalid args: {args}")
            else:
                self.w = len(_ref[0])
                self.h = len(_ref)

            self.__version__ = 1.2
            self._buffer = f"{PostscriptPy.__WATERMARKS[self.__version__]}\n" \
                           f"%!PS-Adobe-3.0 EPSF-3.0\n" \
                           f"%%BoundingBox: 0 0 {self.w} {self.h}\n" \
                           f"{PostscriptPy.__BGST}\n" \
                           f"{PostscriptPy.__BGEND}\n"
            self._frame = ""

        self._defined = set()
        self._color = None
        self._ref = _ref

    def __str__(self):
        return self._buffer

    def out(self, fp=None):

        if fp is None:
            n = 0
            for p in self.__path_out.iterdir():
                if p.is_file() and p.name.startswith("pspy") and p.name.endswith(".eps"):
                    n = max(n, int(p.name.split(".")[0][4:]))
            fp = str(self.__path_out.joinpath(f"pspy{n + 1}.eps"))
        elif not (file := Path(fp)).is_absolute():
            fp = str(self.__path_out.joinpath(file))
            if not fp.endswith(".eps"):
                fp += ".eps"

        if self.__version__ >= 1.2:
            self._buffer += PostscriptPy.__FRST + "\n" + self._frame + PostscriptPy.__FREND + "\n"
        self._buffer += "showpage\n%EOF"
        with open(fp, "w") as psfile:
            psfile.write(self._buffer)

    def draw(self):
        file = str(self.__path_temp.joinpath(f"image.eps"))
        self.out(file)
        match platform.system():
            case "Windows":
                cp = subprocess.run(["gswin64c", "-sDEVICE=display", f"-r{self.w}x{self.h}", file], shell=True)
            case "Darwin":
                cp = subprocess.run(["open", "-a", "Preview", file], shell=True)
            case _:
                raise FileNotFoundError(f"Bad OS for rendering postscript with "
                                        f"default application: {platform.system()}")
        if cp.returncode:
            print(f"Failed to open {file}, exited with {cp.returncode}", file=sys.stderr)

    def convert(self, size: float | tuple[int, int] = 1, mode="png", outfile=None):
        if outfile is None:
            n = 0
            for f in self.__path_out.iterdir():
                if f.is_file() and f.name.endswith(".png") and f.name.startswith("pspy"):
                    n = max(n, int(f.name.split(".")[0][4:]))
            outfile = str(self.__path_out.joinpath(f"pspy{n + 1}.png").absolute())

        path_self = str(self.__path_temp.joinpath("convert_temp.eps").absolute())

        # Don't include watermark
        with open(path_self, "w") as out:
            out.write(self._buffer[self._buffer.index("\n") + 1:] + "showpage\n%EOF")

        match platform.system():
            case "Windows":
                gs = "gswin64c"
            case "Darwin":
                gs = "gs"
            case _:
                raise FileNotFoundError(f"Bad OS for rendering postscript with "
                                        f"default application: {platform.system()}")

        cmd = [gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop"]
        if isinstance(size, tuple):
            cmd.append(f"-r{size[0]}x{size[1]}")
        else:
            cmd.append(f"-r{self.w * size}x{self.h * size}")

        match mode:
            case "png":
                cmd.append("-sDEVICE=pngalpha")
            case "jpg" | "jpeg":
                cmd.append("-sDEVICE=jpg")
            case _:
                raise ValueError(f"Unrecognized image format: {mode}")

        cmd.append(f"-sOutputFile={outfile}")
        cmd.append(path_self)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)

    @property
    def width(self):
        return self.w

    @property
    def height(self):
        return self.h

    @classmethod
    def stdcanvas(cls, w: int, h: int, *colors) -> 'PostscriptPy':
        """
        Creates and returns a PostscriptPy object with stdline configuration and
        the background color set
        """
        eps = PostscriptPy(w, h)
        eps.setstdline()
        if colors:
            eps.background(*colors)
        return eps

    @classmethod
    def join(cls, file1, file2, *files):
        buffer = ""
        for f in (file1, file2, *files):
            if not buffer:
                with open(str(f), "r") as epsfile:
                    buffer += epsfile.read()[:-1]
            else:
                with open(str(f), "r") as epsfile:
                    buffer += epsfile.read()[3:-1]
        n = 0
        for f in PostscriptPy.__path_out.iterdir():
            if f.is_file() and f.name.startswith("joined"):
                n = max(n, int(f.name.split(".")[0][6:]))

        file = str(PostscriptPy.__path_out.joinpath(f"joined{n + 1}.eps"))
        with open(file, "w") as outfile:
            outfile.write(buffer)

    @classmethod
    def gradiant_up(cls, file1, file2, **kwargs):
        # Computes gradiant from entirely file1 to almost full blend of file1 and file2
        grad = 0.0
        while grad < 1.0:
            cls.from_image_blend(file1, file2, grad=grad, **kwargs).out()
            grad += 0.1

    @classmethod
    def gradiant_down(cls, file1, file2, **kwargs):
        # Computes gradiant from full blend file1 and file2 to full file1
        grad = 1.0
        while grad > 0.0:
            cls.from_image_blend(file1, file2, grad=grad, **kwargs).out()
            grad -= 0.1

    @classmethod
    def blend(cls, file1, file2, *files, **kwargs):
        files = [file1, file2, *files]
        n_files = len(files)
        for i in range(n_files):
            cls.gradiant_up(files[i], files[(i + 1) % n_files], **kwargs)
            cls.gradiant_down(files[(i + 1) % n_files], files[i], **kwargs)

    @staticmethod
    def from_image(fp: str):
        return PostscriptPy(_ref=read_image(fp))

    @staticmethod
    def from_image_flexible(
            file: str,
            *,
            k: int = 20,
            gray: bool = False,
            neg: bool = False,
            rndm: bool = False,
            background: tuple[int, int, int] = (255, 255, 255),
            noise: PerlinNoise = None
    ):

        pixels = read_image(file)

        # Function used for creating a function of a line for sides of hexagon
        point_slope_y = lambda m, _x, _y: lambda __y: (__y - _y) / m + _x

        width = sqrt(3) * k  # Full width of hexagon
        half = width / 2  # Half width
        k1 = k / 2  # Half the height of hexagon

        vertices = [[], []]  # First 2 rows empty

        x = half  # Center of first hexagon is at (half, k)

        # Width and height of original images
        p_width = len(pixels[0])
        p_height = len(pixels)

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

        eps = PostscriptPy.stdcanvas(w, h, *background)

        for row in vertices:
            for x, y in row:
                dx = half
                dy = k1

                # Left side of hexagon, upper and lower slopes
                upper_left_bound = point_slope_y(dy / dx, x, y + k)
                lower_left_bound = point_slope_y(-dy / dx, x, y - k)

                colors = (0, 0, 0)
                n_colors = 0

                # Iterate the height of the hexagon top->down

                # From top of hex to first vertex
                for j in range(floor(y + k), ceil(y + k1), -1):

                    x3 = upper_left_bound(j)
                    for i in range(ceil(x3), floor(2 * x - x3)):
                        colors = tuple(a + b for a, b in zip(colors, pixels[j][i]))
                        n_colors += 1

                # From first vertex to second vertex, both sides are vertical so no slope
                for j in range(floor(y + k1), ceil(y - k1), -1):
                    for i in range(ceil(x - half), floor(x + half)):
                        colors = tuple(a + b for a, b in zip(colors, pixels[j][i]))
                        n_colors += 1

                # From second vertex to bottom of hex
                for j in range(floor(y - k1), ceil(y - k), -1):
                    x3 = lower_left_bound(j)
                    for i in range(ceil(x3), floor(2 * x - x3)):
                        colors = tuple(a + b for a, b in zip(colors, pixels[j][i]))
                        n_colors += 1

                # Average color
                colors = tuple(a / n_colors for a in colors)

                v = sum(colors) / 765
                if gray:
                    colors = (0, 0, 0)
                elif noise is not None:
                    v = noise((x / w, y / h)) + 0.5
                elif rndm:
                    v = uniform(0, 1)
                elif not neg:
                    v = 1 - v

                y1 = v * 3 * k / 4
                x1 = v * half / 2
                x2 = v * half

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

    @staticmethod
    def from_image_blend(
            file1: str,
            file2: str,
            *,
            grad: float = 1.0,
            k: int = 20,
            neg: bool = False,
            background: tuple[int, int, int] = (255, 255, 255),
            noise: PerlinNoise = None
    ):
        """
        Creates an image constructed of flexible 6-pointed stars that are colored
        the combination of the images from file1 and file2, the ratio of combination
        being generated by Perlin noise.

        The grad argument allows for the user to choose
        how much of image 2 contributes to the final iamge. A grad value of
        1.0 (default) means that the final color is influenced only by the
        noise function, while a grad value of 0.0 means that the final
        color is entirely image 1, regardless of noise. Grad 0 is equivalent
        to running PostscriptPy.from_image_flexible()

        :param file1: first image
        :param file2: second image
        :param grad: gradiant of how much of file2 to show
        :param k: size of side of hexagon
        :param neg: increase size for lighter color instead of darker
        :param background: background color
        :param noise: PerlinNoise object
        :return: PostscriptPy picture
        """

        pixels1 = read_image(file1)
        pixels2 = read_image(file2)

        if shape(pixels1) != shape(pixels2):
            raise ValueError("Images are not the same shape!")

        # Function used for creating a function of a line for sides of hexagon
        point_slope_y = lambda m, _x, _y: lambda __y: (__y - _y) / m + _x

        width = sqrt(3) * k  # Full width of hexagon
        half = width / 2  # Half width
        k1 = k / 2  # Half the height of hexagon

        vertices = [[], []]  # First 2 rows empty

        x = half  # Center of first hexagon is at (half, k)

        # Width and height of original images
        p_width = len(pixels1[0])
        p_height = len(pixels1)

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

                # Noise for how much of image 2 to include, noise function returns between -0.5 and 0.5
                v1 = (noise((x / w, y / h)) + 0.5) * grad

                colors = tuple(a * (1 - v1) + b * v1 for a, b in zip(colors1, colors2))

                v2 = sum(colors) / 765
                if not neg:
                    v2 = 1 - v2

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

        return eps

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

        # Two values of k that are used a lot and therefore pre-computed and stored
        k1 = k * k
        k2 = k / 2

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
                colors = tuple(x // k1 for x in colors)

                if use_saddle:
                    size = saddle(i, j)
                elif rndm:
                    # Assign random size that is from k-size to k
                    size = randrange(1, k)
                elif gray:
                    # Set color to be black and size to correlate to brightness value
                    colors = (0, 0, 0)
                    size = round(k * ((255 - sum(colors) / 3) / 255), 3)
                elif neg:
                    # If negate, correlate larger with lighter colors
                    size = round((sum(colors) / (3 * 255)) * k, 3)
                else:
                    # Otherwise darker colors means larger
                    size = round(((255 - (sum(colors) / 3)) / 255) * k, 3)

                self.fill_square(*PostscriptPy.center_square(ik + k2, h - (jk + k2), size), *colors)

    def curve_fractal(self, depth: int = 2, f_h: float = 0.5, f_l: float = 0.5):

        def get_intercepts(fx: Callable, fy: Callable, U=None, D=0, L=0, R=None):
            if U is None:
                U = self.h
            if R is None:
                R = self.w

            # If line intersects the lower bound within the right and left bounds
            if R >= fy(D) >= L:
                _p1 = (fy(D), D)

                # Find second intersection among upper, right, and left bounds
                if R >= fy(U) >= L:
                    _p2 = (fy(U), U)
                elif U >= fx(R) >= D:
                    _p2 = (R, fx(R))
                else:
                    _p2 = (L, fx(L))

            # If line intersects left bound between upper and lower
            elif U >= fx(L) >= D:
                _p1 = (L, fx(L))

                # Find second intersection among upper and right bounds since it wasn't down
                if R >= fy(U) >= L:
                    _p2 = (fy(U), U)
                else:
                    _p2 = (R, fx(R))

            # Since the line MUST intersect two sides, if it didn't intersect either left or lower,
            # it must be upper and right
            else:
                _p1 = (R, fx(R))
                _p2 = (fy(U), U)

            return _p1, _p2

        def get_bounding(_p1, _p2, _theta, U=None, D=0, L=0, R=None):

            if U is None:
                U = self.h
            if R is None:
                R = self.w

            from math import pi

            # If angle is between 0° and 135°, then the largest y value is considered the 'first' point
            if _theta < 3 * pi / 4:
                if _p1[1] < _p2[1]:
                    _p1, _p2 = _p2[:], _p1[:]

            # If angle is between 135° and 225°, then 'first' is lowest x
            elif _theta < 5 * pi / 4:
                if _p1[0] > _p2[0]:
                    _p1, _p2 = _p2[:], _p1[:]

            # If angle between 225° and 305°, then 'first' is lowest y
            elif _theta < 7 * pi / 4:
                if _p1[1] > _p2[1]:
                    _p1, _p2 = _p2[:], _p1[:]

            # Otherwise, if angle between 305° and 360°, then 'first' is largest x
            else:
                if _p1[0] < _p2[0]:
                    _p1, _p2 = _p2[:], _p1[:]

            # Vertices of bounding box
            vertices = [(L, D), (L, U), (R, U), (R, D)]

            # Find where p1 fits in, in cyclic order
            if _p1[1] == D:
                i1 = 4
            elif _p1[0] == L:
                i1 = 1
            elif _p1[1] == U:
                i1 = 2
            else:
                i1 = 3

            # Find where p2 fits in, in cyclic order
            if _p2[1] == D:
                i2 = 4
            elif _p2[0] == L:
                i2 = 1
            elif _p2[1] == U:
                i2 = 2
            else:
                i2 = 3

            if i2 < i1:
                vertices.insert(i1, _p1)
                vertices.insert(i2, _p2)
                i1 += 1
            else:
                vertices.insert(i2, _p2)
                vertices.insert(i1, _p1)
                i2 += 1

            points = []

            k = i1
            while k != i2:
                points.append(vertices[k])
                k = (k + 1) % 6

            points.append(vertices[i2])
            return points

        from math import pi

        pi2 = 2 * pi

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
            theta = (theta + d_theta) % pi2

            f_x = point_slope_x(tan(theta), *point)
            f_y = point_slope_y(tan(theta), *point)

            p1, p2 = get_intercepts(f_x, f_y)
            point = ((p1[0] + p2[0]) * f_l, (p1[1] + p2[1]) * f_l)

            polys.appendleft(get_bounding(p1, p2, theta))

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

    def frame(self, w: float = 10, *color):
        if self.__version__ >= 1.2:
            if color:
                self.color(*color)
            if len(set(color)) == 1:
                self._frame = f"{round(color[0] / 255, 3)} setgray\n"
            elif color:
                r, g, b = tuple(round(a / 255, 3) for a in color)
                self._frame = f"{r} {g} {b} setrgbcolor\n"
            else:
                raise ValueError("Frame must have color!")

            # This should not depend on fill rect being defined
            self._frame += (
                    f"newpath\n" +
                    f"0 0 moveto\n" +
                    f"0 {self.h} lineto\n" +
                    f"{self.w} {self.h} lineto\n" +
                    f"{self.w} 0 lineto\n" +
                    f"{self.w - w} 0 lineto\n" +
                    f"{self.w - w} {self.h - w} lineto\n" +
                    f"{w} {self.h - w} lineto\n" +
                    f"{w} {w} lineto\n" +
                    f"{self.w - w} {w} lineto\n" +
                    f"{self.w - w} 0 lineto\n" +
                    f"closepath\n" +
                    f"fill\n"
            )
        else:
            self.frame_layer(w, *color)

    def frame_layer(self, w: float = 10, *color):
        self.fill_rect(0, 0, w, self.h, *color)             # Left border
        self.fill_rect(0, self.h - w, self.w, w, *color)    # Top border
        self.fill_rect(self.w - w, 0, w, self.h, *color)    # Right border
        self.fill_rect(0, 0, self.w, w, *color)             # Bottom border

    def background(self, r, g: float = None, b: float = None):
        if self.__version__ >= 1.1:
            lines = self._buffer.split("\n")
            start = lines.index(PostscriptPy.__BGST)
            end = lines.index(PostscriptPy.__BGEND)
            self._buffer = "\n".join(lines[:start + 1]) + "\n"
            self.color(r, g, b)

            # This should not depend on fill rect being defined
            self._buffer += ("newpath\n" +
                             "0 0 moveto\n" +
                             f"{self.w} 0 lineto\n" +
                             f"{self.w} {self.h} lineto\n" +
                             f"0 {self.h} lineto\n" +
                             "closepath\n" +
                             "fill\n")
            self._buffer += "\n".join(lines[end:]) + "\n"
        else:
            self.fill_rect(0, 0, self.w, self.h, r, g, b)

    def translate(self, x, y):
        self._buffer += f"{x} {y} translate\n"

    def setstdline(self, width: float = 2):
        """
        Sets the line settings to 'standard' of linejoin 1, linecap 1, and linewidth 2
        unless a different width parameter is provided.

        :param width: optional width
        :return: None
        """
        self.setlinejoin()
        self.setlinecap()
        self.setlinewidth(width)

    def setlinejoin(self, value: int = 1):
        self._buffer += f"{value} setlinejoin\n"

    def setlinecap(self, value: int = 1):
        self._buffer += f"{value} setlinecap\n"

    def setlinewidth(self, value):
        self._buffer += f"{value} setlinewidth\n"

    def color(self, r: float, g: float = None, b: float = None):
        if g is None or r == g == b:
            gray = round(r / 255, 3)
            if gray != self._color:
                self._buffer += f"{gray} setgray\n"
                self._color = gray
        else:
            r, g, b = tuple(round(a / 255, 3) for a in (r, g, b))
            if (r, g, b) != self._color:
                self._buffer += f"{r} {g} {b} setrgbcolor\n"
                self._color = (r, g, b)

    def newpath(self, *args):
        self._buffer += "newpath\n"
        if args:
            if len(args) == 2:
                self._buffer += f"{args[0]} {args[1]} moveto\n"
            else:
                raise ValueError(f"Invalid {PostscriptPy.newpath.__name__} args: {args}")

    def rotate(self, theta):
        self._buffer += f"{theta} rotate\n"

    def draw_line(
            self,
            x: float, y: float,  # Initial coordinates (required)
            x2: float = None, y2: float = None,  # Second coordinates (optional)
            *color,
            length: int = None, theta: float = None  # Length and direction (optional)
    ):
        """
        Draws a line starting at (x, y) and either ending at (x2, y2) or calculating
        a second pair of coordinates using length and angle and drawing there. Either
        the second pair of coordinates or length AND angle are required.

        :param x: starting x
        :param y: starting y
        :param x2: optional end x
        :param y2: optional end y
        :param color: color of line
        :param length: optional length of line
        :param theta: optional angle of line
        """
        if isinstance(length, int) and isinstance(theta, (int, float)):
            x2 = x + length * cos(radians(theta))
            y2 = y + length * sin(radians(theta))
        if x2 is not None and y2 is not None:
            if color:
                self.color(*color)
            self._buffer += f"newpath\n" \
                            f"{x} {y} moveto\n" \
                            f"{x2} {y2} lineto\n" \
                            f"closepath\n" \
                            f"stroke\n"
        else:
            raise ValueError("Must provide either end desination or length and angle")

    def draw_rect(self, x: float, y: float, w: float, h: float, *color):
        if PostscriptPy.EPS.RECT not in self._defined:
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
            self._defined.add(PostscriptPy.EPS.RECT)

        if color:
            self.color(*color)
        self._buffer += f"{x} {y} {w} {h} draw_rect\n"

    def fill_rect(self, x: float, y: float, w: float, h: float, *color):
        if PostscriptPy.EPS.RECT_FILL not in self._defined:
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
            self._defined.add(PostscriptPy.EPS.RECT_FILL)

        if color:
            self.color(*color)
        self._buffer += f"{x} {y} {w} {h} fill_rect\n"

    def draw_square(self, x: float, y: float, w: float, *color):
        if PostscriptPy.EPS.SQUARE not in self._defined:
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
            self._defined.add(PostscriptPy.EPS.SQUARE)

        if color:
            self.color(*color)
        self._buffer += f"{x} {y} {w} draw_square\n"

    def fill_square(self, x: float, y: float, w: float, *color):
        if PostscriptPy.EPS.SQUARE_FILL not in self._defined:
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
            self._defined.add(PostscriptPy.EPS.SQUARE_FILL)

        if color:
            self.color(*color)
        self._buffer += f"{x} {y} {w} fill_square\n"

    def __poly(self, vertices: Sequence, *color):
        if color:
            self.color(*color)
        self._buffer += (
                f"newpath\n" +
                f"{vertices[0][0]} {vertices[0][1]} moveto\n"
        )
        for i in range(1, len(vertices)):
            self._buffer += f"{vertices[i][0]} {vertices[i][1]} lineto\n"
        self._buffer += "closepath\n"

    def draw_poly(self, vertices: Sequence, *color):
        self.__poly(vertices, *color)
        self._buffer += "stroke\n"

    def fill_poly(self, vertices: Sequence, *color):
        self.__poly(vertices, *color)
        self._buffer += "fill\n"

    @staticmethod
    def center_rect(x: float, y: float, w: int, h: int) -> tuple[float, float, int, int]:
        """Helper function to give top left coordinates of rectangle with center at (x, y)"""
        return x - (w / 2), y - (h / 2), w, h

    @staticmethod
    def center_square(x: float, y: float, w: int) -> tuple[float, float, int]:
        """Helper function to give top left coordinates of square with center at (x, y)"""
        return x - (w / 2), y - (w / 2), w


def load_postscript_py(file: str) -> PostscriptPy:
    if not file.endswith(".eps"):
        raise ValueError(f"{file} is not a valid postscript file!")
    else:
        with open(file, "r") as f:
            buffer = f.read()
            if buffer.startswith(PostscriptPy.WATERMARK):
                return PostscriptPy(_load=True, _buffer=buffer)
            else:
                raise ValueError(f"Not a valid {PostscriptPy.__class__.__name__} file!")
