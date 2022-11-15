from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import label2rgb

# open rover one image
img_init0 = Image.open('images/0073.tif')
img_init1 = Image.open('images/0174.tif')
img_init2 = Image.open('images/0617.tif')
img_init3 = Image.open('images/1059.tif')

# save image as numpy array
img0 = np.asarray(img_init0)
img1 = np.asarray(img_init1)
img2 = np.asarray(img_init2)
img3 = np.asarray(img_init3)

# convert image to HSV
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

# show original image
plt.figure(1)
plt.imshow(img3)
plt.axis('off')
plt.show()