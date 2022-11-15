import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
import cv2

# Input data
# open rover one image
img_init = Image.open('images/0073.tif')

# save image as numpy array
img = np.asarray(img_init)

# convert image to HSV
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(morphology.remove_small_objects(lum < 0.7, 500), 500)

mask = morphology.opening(mask, morphology.disk(3))

# SLIC result
slic = segmentation.slic(img, n_segments=100, start_label=1)

# maskSLIC result
m_slic = segmentation.slic(img, n_segments=100, mask=mask, start_label=1)

# Display result
fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
ax1, ax2, ax3, ax4 = ax_arr.ravel()

ax1.imshow(img)
ax1.set_title('Original image')

ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')

ax3.imshow(segmentation.mark_boundaries(img, slic))
ax3.contour(mask, colors='red', linewidths=1)
ax3.set_title('SLIC')

ax4.imshow(segmentation.mark_boundaries(img, m_slic))
ax4.contour(mask, colors='red', linewidths=1)
ax4.set_title('maskSLIC')

for ax in ax_arr.ravel():
    ax.set_axis_off()

plt.tight_layout()
plt.show()