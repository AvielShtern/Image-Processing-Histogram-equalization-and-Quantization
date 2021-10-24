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
    """
    reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: either 1 or 2 defining whether the output should be a grayscale image (1) or an RGB image (2).
    :return: image is represented by a matrix of type np.float64 with intensities (either grayscale or RGB channel
             intensities) normalized to the range [0, 1].
    """
    rgb_or_gray_scale_image = imread(filename).astype(np.float64) / 255
    if representation == RGB_REPRESENTATION or (
            representation == GRAYSCALE_REPRESENTATION and rgb_or_gray_scale_image.ndim == 2):
        return rgb_or_gray_scale_image
    elif representation == GRAYSCALE_REPRESENTATION and rgb_or_gray_scale_image.ndim == 3:
        return rgb2gray(rgb_or_gray_scale_image)


def imdisplay(filename, representation):
    """
     open a new figure and display the loaded image in the converted representation
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: either 1 or 2 defining whether the display should be a grayscale image (1) or an RGB image (2).
    :return: None
    """
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


def quantize(im_orig, n_quant, n_iter):
    # assume im_origin in grayscale
    im_to_work = np.around(im_orig * MAX_LEVEL).astype(np.uint32)
    num_pixels = im_to_work.shape[0] * im_to_work.shape[1]
    original_bins = np.histogram(im_orig, bins=MAX_LEVEL + 1, range=(MIN_LEVEL, MAX_LEVEL + 1))[0]

    cbins = np.cumsum(original_bins)
    cbins_weighted = np.cumsum(original_bins * np.arange(MAX_LEVEL + 1))

    num_of_pixel_in_segment = (num_pixels / n_quant)  # for initial step
    num_of_pixel_in_segment_in_cbins = np.arange(n_quant + 1) * num_of_pixel_in_segment
    initial_Z = np.digitize(num_of_pixel_in_segment_in_cbins, cbins, right=True).astype(np.uint32)

    initial_Q = [cbins_weighted[initial_Z[1]] / (cbins[initial_Z[1]] + 1)] + \
                [(cbins_weighted[initial_Z[i + 1]] - cbins_weighted[initial_Z[i]]) / \
                 (cbins[initial_Z[i + 1]] - cbins[initial_Z[i]] + 1) for i in range(1, n_quant)]
    # loop for n_quant allowd ("Specific Guidelines" 3)
    # divide by 0 + 1

    curr_Z = initial_Z
    curr_Q = initial_Q
    error = []
    for i in range(n_iter):
        candidate_to_Z = np.array(
            [MIN_LEVEL - 1] + [(curr_Q[i] - curr_Q[i - 1]) / 2 for i in range(1, n_quant)] + [MAX_LEVEL]).astype(
            np.uint32)
        if np.all(candidate_to_Z == curr_Z):
            break
        curr_Z = candidate_to_Z
        curr_Q = [cbins_weighted[curr_Z[1]] / (cbins[curr_Z[1]] + 1)] + \
                 [((cbins_weighted[initial_Z[i + 1]] - cbins_weighted[initial_Z[i]]) / (
                         cbins[initial_Z[i + 1]] - cbins[initial_Z[i]] + 1)) for i in range(1, n_quant)]
        # divide by zero +1

        error.append(np.sum(np.power(np.repeat(curr_Q, np.diff(curr_Z)) - np.arange(MAX_LEVEL + 1), 2) * original_bins))
        # sum of (qi - g)^2 * h(g)

    look_up_table = np.repeat(curr_Q, np.diff(curr_Z))
    return [look_up_table[im_to_work], error]


if __name__ == '__main__':
    print(read_image("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/image2.png", 2))
