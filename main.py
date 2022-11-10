from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
ECE597IP PROJECT 1 
Author: Calista Greenway
This program performs image segmentation on mars rover images by clustering and color space transformations.
"""

# open rover images
img0 = Image.open('images/0073.tif')
img1 = Image.open('images/0174.tif')
img2 = Image.open('images/0617.tif')
img3 = Image.open('images/1059.tif')
img = [img0, img1, img2, img3]

# save images as .jpg
for i in range(4):
    data = np.asarray(img[i])
    img[i] = Image.fromarray(data)
    img[i].save("img" + str(i) + ".jpg")

# convert images to RGB
img_rgb = []
for i in range(4):
    rgb = cv2.cvtColor(cv2.imread("img" + str(i) + ".jpg"),  cv2.COLOR_BGR2RGB)
    img_rgb.append(rgb)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
img_pixel_val = []
for i in range(4):
    img_pixel_val = img_rgb[i].reshape((-1, 3))
    img_pixel_val = np.float32(img_pixel_val)

# number of clusters (k)
k = 3
