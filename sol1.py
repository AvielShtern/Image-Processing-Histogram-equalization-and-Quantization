import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

GRAYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2
Y_POSITION = 0
MIN_LEVEL = 0
MAX_LEVEL = 255
DIM_IMAGE_GRAY = 2
DIM_IMAGE_RGB = 3


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
    rgb_or_gray_scale_image = imread(filename).astype(np.float64) / MAX_LEVEL
    if representation == RGB_REPRESENTATION or (
            representation == GRAYSCALE_REPRESENTATION and rgb_or_gray_scale_image.ndim == DIM_IMAGE_GRAY):
        return rgb_or_gray_scale_image
    elif representation == GRAYSCALE_REPRESENTATION and rgb_or_gray_scale_image.ndim == DIM_IMAGE_RGB:
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
    """
    transform an RGB image into the YIQ color space
    :param imRGB: height×width×3 np.float64 matrices in the [0, 1] range. the red channel is encoded in imRGB[:,:,0],
                  the green in imRGB[:,:,1], and the blue in imRGB[:,:,2]
    :return: imYIQ (height×width×3 np.float64 matrices in the [0, 1] range) imYIQ[:,:,0] encodes the luminance channel Y,
             imYIQ[:,:,1] encodes I, and imYIQ[:,:,2] encodes Q
    """
    return imRGB @ RGB2YIQ.T


def yiq2rgb(imYIQ):
    """
    transform an YIQ image into the RGB color space
    :param imYIQ: height×width×3 np.float64 matrices in the [-1, 1] range (Y in [0,1]). imYIQ[:,:,0] encodes the luminance channel Y,
           imYIQ[:,:,1] encodes I, and imYIQ[:,:,2] encodes Q
    :return: imRGB (height×width×3 np.float64 matrices in the [0, 1] range.) the red channel is encoded in imRGB[:,:,0],
             the green in imRGB[:,:,1], and the blue in imRGB[:,:,2]
    """
    return imYIQ @ YIQ2RGB.T


def histogram_equalize(im_orig):
    """
    performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: grayscale or RGB float64 image with values in [0, 1].
    :return: list [im_eq, hist_orig, hist_eq] where
            im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
            hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    type_of_im, yiq_img, im_to_work = cases_rgb_grayscale(im_orig)

    hist_origin = np.histogram(im_to_work, bins=MAX_LEVEL + 1, range=(MIN_LEVEL, MAX_LEVEL + 1))[0]
    C = np.cumsum(hist_origin).astype(np.float64)
    M = (np.argwhere(C > 0))[0][0]  # M be the first gray level for which C(M) != 0
    T = np.around(MAX_LEVEL * ((C - C[M]) / (C[MAX_LEVEL] - C[M]))).astype(np.int32)  # lookup table

    if type_of_im == RGB_REPRESENTATION:
        yiq_img[:, :, Y_POSITION] = (T[im_to_work].astype(np.float64) / MAX_LEVEL)
    im_eq = (T[im_to_work].astype(np.float64) / MAX_LEVEL) if type_of_im == GRAYSCALE_REPRESENTATION else yiq2rgb(
        yiq_img)
    hist_eq = np.histogram(T[im_to_work], bins=MAX_LEVEL + 1, range=(MIN_LEVEL, MAX_LEVEL + 1))[0]

    return [im_eq, hist_origin, hist_eq]


def cases_rgb_grayscale(im_orig):
    """
    helper for histogram_equalize and quantize for cases of grayscale/rgb
    :param im_origin: grayscale or RGB float64 image with values in [0, 1].
    :return: tuple (type_of_im, yiq_img, im_to_work) where:
            type_of_im = 1 if the image in grayscale (im.ndim = 2) else (im.ndim = 3) type_of_im = 2
            yiq_img = if type_of_im = 2 the yiq convert. else None
            im_to_work is the image to work (gray scale or rgb as yiq) where im_to_work.ndim = 2, the range is [0,255]
            and type is int32
    """
    type_of_im = GRAYSCALE_REPRESENTATION if im_orig.ndim == DIM_IMAGE_GRAY else RGB_REPRESENTATION
    yiq_img = rgb2yiq(im_orig) if type_of_im == RGB_REPRESENTATION else None
    rgb_or_grayscale = im_orig if type_of_im == GRAYSCALE_REPRESENTATION else yiq_img[:, :, Y_POSITION]
    im_to_work = np.around(rgb_or_grayscale * MAX_LEVEL).astype(np.int32)
    return type_of_im, yiq_img, im_to_work


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image.
    :param im_orig:  grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of intensities the output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: a list [im_quant, error] where:
                im_quant - is the quantized output image. (float64 image with values in [0, 1]).
                error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration
                of the quantization procedure.
    """
    type_of_im, yiq_img, im_to_work = cases_rgb_grayscale(im_orig)
    total_num_pixels = im_to_work.shape[0] * im_to_work.shape[1]
    original_bins = np.histogram(im_to_work, bins=MAX_LEVEL + 1, range=(MIN_LEVEL, MAX_LEVEL + 1))[0].astype(np.int32)

    cbins = np.cumsum(original_bins)
    cbins_weighted = np.cumsum(original_bins * np.arange(MAX_LEVEL + 1))  # g*h(g)

    num_of_pixel_in_segment = (total_num_pixels / n_quant)  # for initial step

    initial_Z = np.array(
        [MIN_LEVEL - 1] + [np.argmin(np.abs(cbins - (i * num_of_pixel_in_segment))) for i in range(1, n_quant)] + [MAX_LEVEL])
                                                             # loop for n_quant allowd ("Specific Guidelines" 3)

    initial_Q = np.array([cbins_weighted[initial_Z[1]] / cbins[initial_Z[1]]] + \
                         [(cbins_weighted[initial_Z[i + 1]] - cbins_weighted[initial_Z[i]]) / \
                          (cbins[initial_Z[i + 1]] - cbins[initial_Z[i]]) for i in range(1, n_quant)])
                                                 # loop for n_quant allowd ("Specific Guidelines" 3)

    curr_Z = initial_Z
    curr_Q = initial_Q
    error = []
    for i in range(n_iter):
        candidate_to_Z = np.array(
            [MIN_LEVEL - 1] + [(curr_Q[i] + curr_Q[i - 1]) / 2 for i in range(1, n_quant)] + [MAX_LEVEL]).astype(np.int32)
                                                                # loop for n_quant allowd ("Specific Guidelines" 3)
        if np.all(candidate_to_Z == curr_Z):
            break
        curr_Z = candidate_to_Z
        curr_Q = np.array([cbins_weighted[curr_Z[1]] / (cbins[curr_Z[1]])] + \
                          [((cbins_weighted[curr_Z[i + 1]] - cbins_weighted[curr_Z[i]]) / (
                                  cbins[curr_Z[i + 1]] - cbins[curr_Z[i]])) for i in range(1, n_quant)])

        error.append(np.sum(np.power(np.repeat(curr_Q, np.diff(curr_Z)) - np.arange(MAX_LEVEL + 1), 2) * original_bins))
        # sum of (qi - g)^2 * h(g)

    look_up_table = np.repeat(np.around(curr_Q).astype(np.int32), np.diff(curr_Z)).astype(np.uint32)
    if type_of_im == RGB_REPRESENTATION:
        yiq_img[:, :, Y_POSITION] = (look_up_table[im_to_work].astype(np.float64) / MAX_LEVEL)
        return [yiq2rgb(yiq_img), error]

    im_quant = (look_up_table[im_to_work].astype(np.float64) / MAX_LEVEL)
    return [im_quant, np.array(error)]


def quantize_rgb(im_orig, n_quant):
    """
     perform quantization for full color images
     In fact, this is a "Clustering" problem (hint was in the instruction file) we are interested in partitioning our
     data (points in R3) into groups When the "centroid" of each cluster (group) will be the color (r,g,b) to which we
     will map all the points in that group. The loss function is pretty much the same as what we learned in class
     (Euclidean distance from the center and also the weight (h) part of the equation because it will be the "amount of
     points" there are of each color)
     for this problem I using K-means algorithm/// that we learn in iml course (lest year) And for that I use the
     library "scikit-learn".
    :param im_orig: grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: the number of intensities your output im_quant image should have.
    :return: the quantized image im_quant(float64 image with values in [0, 1]).
    """
    from sklearn.cluster import KMeans
    num_of_iteretion = 300  # Number of iterations that the model will run (unless there was a convergence)
    rows = im_orig.shape[0]
    cols = im_orig.shape[1]
    kmeans = KMeans(n_clusters=n_quant, max_iter=num_of_iteretion, random_state=0).fit(im_orig.reshape(rows * cols, 3))
    return kmeans.cluster_centers_[kmeans.labels_].reshape(rows, cols, 3)