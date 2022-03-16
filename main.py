from postscript import *
from PIL import Image
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

    gradiant_up(dancer, pixels)
    gradiant_down(pixels, dancer)
    gradiant_up(pixels, sword)
    gradiant_down(sword, pixels)
    gradiant_up(sword, dancer)
    gradiant_down(dancer, sword)


def gif():
    frames = []
    for i in range(1, 67):
        img = Image.open(f"tempgif/pspy{i}.png")  # type: Image.Image
        # img = img.resize((img.width // 8, img.height // 8), Image.ANTIALIAS)
        # img.save(f"tempgif/pspy{i}.png")
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
    # eps = PostscriptPy.from_image_blend("mbdtf4.jpeg", "mbdtf5.png", k=8, grad=1.0, noise=noise, background=(0, 0, 0))
    # eps.frame(15, 0.861 * 255, 0.136 * 255, 0.237 * 255)
    # eps.draw()
    # blend()
    # out = PostscriptPy.source.joinpath("out")
    # to_join = []
    # j = 1
    # while (file := out.joinpath(f"pspy{j}.eps")).is_file():
    #     to_join.append(str(file))
    #     j += 1
    # PostscriptPy.join(*to_join)
    # for i in range(1, 67):
    #     eps = load_postscript_py(f"out/pspy{i}.eps")
    #     eps.convert(2, outfile=f"tempgif/pspy{i}.png")
    same = set()
    temp = Path(__file__).parent.joinpath("tempgif")
    for file in temp.iterdir():
        if file.is_file():
            for file2 in temp.iterdir():
                if file2.is_file() and file.name != file2.name:
                    if not system(f"diff -q {file} {file2} > /dev/null"):
                        same.add(tuple(sorted((file.name, file2.name))))

    print(same)
    # img = Image.open("./out/pspy1.eps")     # type: Image.Image
    # jpg = img.convert()
    # jpg.save("temp1.png", lossless=True)
    # eps = PostscriptPy.from_image_flexible("mbdtf3.png", bckgrnd=(0, 0, 0))
    # eps.draw()
