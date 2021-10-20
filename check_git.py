import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgba2rgb

GRAYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2

x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))

RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                    [0.569, -0.275, -0.321],
                    [0.212, -0.523, 0.311]]).astype(np.float64)
YIQ2RGB = np.linalg.inv(RGB2YIQ)


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


def rgb2yiq(imRGB):
    return imRGB @ RGB2YIQ.T


def yiq2rgb(imYIQ):
    return imYIQ @ YIQ2RGB


a = np.array([[1, 2, 3], [1, 4, 5]])
b = a + 2
c = np.dstack((a, b))
d = c @ np.array([[1, 2],
                  [3, 4]])
print(c.shape)
print(c)
print(d.shape)
print(d)

# if __name__ == '__main__':
# imdisplay("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/image2.png", 2)
# imdisplay("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/image2.png", 1)
