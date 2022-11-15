import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
ECE597IP PROJECT 1
Author: Calista Greenway
"""

# open rover image(s)
img_init = Image.open('images/0174.tif')

# save image(s) as numpy array
img = np.asarray(img_init)

# smooth image
from skimage import filters

# apply median filter of szie 5x5
smooth_image = filters.median(img)

# plot smoothed image
f, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(img)
ax1.imshow(smooth_image)
plt.show()

# find edges
from skimage import feature
data = np.array(img)
data = data.astype('float32')
edges = feature.canny(data, sigma=2)
edges = np.uint8(feature.canny(data, sigma=1, ) * 255)
plt.imshow(edges)
plt.show()
