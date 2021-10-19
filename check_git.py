import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt

x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))





