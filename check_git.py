import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgba2rgb

GRAYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2

x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))


def read_image(filename, representation):
    rgb_image = imread(filename).astype(np.float64) / 255
    return rgb_image if representation == RGB_REPRESENTATION \
        else rgb2gray(rgb_image) if representation == GRAYSCALE_REPRESENTATION \
        else -1


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    print(im.shape)
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    imdisplay("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/image2.png", 2)
    imdisplay("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/image2.png", 1)
