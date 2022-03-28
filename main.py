from src.postscript import *
from PIL import Image
from perlin_noise import PerlinNoise
import subprocess
from numpy import shape


noise = PerlinNoise(octaves=4, seed=1)


def gradiant_up(file1, file2):
    # Computes gradiant from entirely file1 to almost full blend of file1 and file2
    grad = 0.0
    while grad < 1.0:
        PostscriptPy.from_image_blend(file1, file2, k=8, grad=grad, noise=noise, background=(0, 0, 0)).out()
        grad += 0.1


def gradiant_down(file1, file2):
    # Computes gradiant from full blend file1 and file2 to full file1
    grad = 1.0
    while grad > 0.0:
        PostscriptPy.from_image_blend(file1, file2, k=8, grad=grad, noise=noise, background=(0, 0, 0)).out()
        grad -= 0.1


def blend():
    dancer = "mbdtf5.png"        # Dancer
    pixels = "mbdtf4.jpeg"       # Pixels
    sword = "mbdtf6.jpg"        # Sword

    gradiant_up(dancer, pixels)         # All of dancer to before half
    gradiant_down(pixels, dancer)       # Half, to before all pixels

    gradiant_up(pixels, sword)          # All pixels, to before half
    gradiant_down(sword, pixels)        # Half, to before all sword

    gradiant_up(sword, dancer)          # All sword, to before half
    gradiant_down(dancer, sword)        # Half, to before all dancer


def gif():
    frames = []
    for i in range(1, 67):
        img = Image.open(f"tempgif/pspy{i}.png")  # type: Image.Image
        frames.append(img)
        print(f"\rLoaded frame {i}", end="")

    frames[0].save('png_to_gif.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=30, loop=0)


if __name__ == "__main__":
    # eps = load_postscript_py("temp/temp.eps")
    # print(eps.__version__)
    # eps.frame(15, 240, 188, 85)
    # eps.background(40, 40, 40)
    # eps.draw()
    # noise = PerlinNoise(octaves=4, seed=1)
    PostscriptPy.blend("mbdtf4.jpeg", "mbdtf5.png", k=8, noise=noise, background=(0, 0, 0))
    out = PostscriptPy.source.joinpath("out")

    
    # to_join = []
    # j = 1
    # while (file := out.joinpath(f"pspy{j}.eps")).is_file():
    #     to_join.append(str(file))
    #     j += 1
    # i = 1
    # for f in file:
    #     eps = load_postscript_py(f)
    #     eps.convert(0.5, outfile=f"tempgif/pspy{i}.png")
    #     i += 1


    # make_gif("tempgif")
    # img = Image.open("./out/pspy1.eps")     # type: Image.Image
    # jpg = img.convert()
    # jpg.save("temp1.png", lossless=True)
    # eps = PostscriptPy.from_image_flexible("mbdtf3.png", bckgrnd=(0, 0, 0))
    # eps.draw()
