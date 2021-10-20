import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

GRAYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2
Y_POSITION = 0
MIN_LEVEL = 0
MAX_LEVEL = 255

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
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.axis("off")
    plt.show()


def rgb2yiq(imRGB):
    return imRGB @ RGB2YIQ.T


def yiq2rgb(imYIQ):
    return imYIQ @ YIQ2RGB.T


def histogram_equalize(im_orig):
    type_of_im = GRAYSCALE_REPRESENTATION if im_orig.ndim - 1 == GRAYSCALE_REPRESENTATION else RGB_REPRESENTATION
    yiq_img = rgb2yiq(im_orig) if type_of_im == RGB_REPRESENTATION else None
    rgb_or_grayscale = im_orig if type_of_im == GRAYSCALE_REPRESENTATION else yiq_img[:, :, Y_POSITION]
    im_to_work = np.around(rgb_or_grayscale * 255).astype(np.uint32)

    hist_origin = np.histogram(im_to_work, bins=MAX_LEVEL + 1, range=(MIN_LEVEL, MAX_LEVEL + 1))[0]
    C = np.cumsum(hist_origin.astype(np.float64))
    M = (np.argwhere(C > 0))[0][0]  # M be the first gray level for which C(M) != 0
    T = np.around(255 * ((C - C[M]) / (C[MAX_LEVEL] - C[M]))).astype(np.uint32)  # lookup table

    if type_of_im == RGB_REPRESENTATION:
        yiq_img[:, :, Y_POSITION] = (T[im_to_work] / 255).astype(np.float64)
    im_eq = (T[im_to_work] / 255).astype(np.float64) if type_of_im == GRAYSCALE_REPRESENTATION else yiq2rgb(yiq_img)
    hist_eq = np.histogram(T[im_to_work], bins=MAX_LEVEL + 1, range=(MIN_LEVEL, MAX_LEVEL + 1))[0]

    return [im_eq, hist_origin, hist_eq]


if __name__ == '__main__':
    # imRGB = read_image("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/image2.png", 2)
    # print(histogram_equalize(imRGB)[2].shape)
