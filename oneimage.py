from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
ECE597IP PROJECT 1 (single image test)
Author: Calista Greenway
"""

# open rover one image
img_init = Image.open('images/0073.tif')

# save image as numpy array
img = np.asarray(img_init)

# show original image
plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")

# convert image to RGB
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

# define stopping criteria
# ...interations reach 100 or clusters move less than 0.2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (k), init wiht random assignment
# "labels" is the cluster label (0 through k)
# "centers" is each centroids value
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(img.shape)

# show segmented image
plt.subplot(1,3,2)
plt.imshow(segmented_image)
plt.title("Segmented Image")

# convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# get threshold image
ret, thresh_img = cv2.threshold(img_grey, 100, 255, cv2.THRESH_BINARY)

# find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)

# show contoured image
plt.subplot(1,3,3)
plt.imshow(img)
plt.title("Contoured Image")

plt.show()
