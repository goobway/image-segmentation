from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import morphology
from skimage import data, segmentation, color
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2lab
from skimage.io import imread, imshow
from skimage.morphology import skeletonize

"""
ECE 597IP FINAL PROJECT 1
Author: Calista Greenway
This program performs image segmentation on mars rover images by clustering and color space transformations.
"""

# open rover images as numpy array (input images)
img0 = np.asarray(Image.open('images/0073.tif'))
img1 = np.asarray(Image.open('images/0174.tif'))
img2 = np.asarray(Image.open('images/0617.tif'))
img3 = np.asarray(Image.open('images/1059.tif'))

# crop images with black boarder
img0 = img0[:, 80:1512]
img2 = img2[:, 80:1536]


def kmeans(img):

    # show original image
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("(a)")
    plt.axis('off')

    # convert image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # reshape the image to a 2D array of pixels and 3 color values
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    # ...iterations reach 100 or clusters move less than 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (k), init with random assignment
    # "labels" is the cluster label (0 through k)
    # "centers" is each centroids value
    k = 6
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)

    # show segmented image
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title("(b)")
    plt.axis('off')

    # convert img to grey
    img_grey = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # get threshold image
    ret, thresh_img = cv2.threshold(img_grey, 100, 255, cv2.THRESH_BINARY)

    # find contours
    contours, hierarchy = cv2.findContours(
        thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 3)

    # show contoured image
    plt.subplot(1, 3, 3)
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("(c)")

    plt.axis('off')
    plt.show()

    # compute a mask
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # lum = color.rgb2gray(img)
    # mask = morphology.remove_small_holes(morphology.remove_small_objects(lum < 0.7, 500), 500)
    # mask = morphology.opening(mask, morphology.disk(3))
    # plt.imshow(mask)
    # plt.axis('off')
    # plt.show()


def meansShift(img):
    # load image
    img = rgb2lab(img)

    # shape of original image
    og_shape = img.shape

    # converting image into array of dimension [nb of pixels in og image, 3]
    # based on r g b intensities
    pixel_values = np.reshape(img, [-1, 3])

    # estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(pixel_values, quantile=0.1, n_samples=200)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # performing meanshift on pixel values
    ms.fit(pixel_values)

    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels = ms.labels_

    # remaining colors after meanshift
    cluster_centers = ms.cluster_centers_

    # finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("# of estimated clusters: %d" % n_clusters_)

    segmented_image = cluster_centers[np.reshape(labels, og_shape[:2])]

    # show segmentation result
    cv2.imshow('Image', segmented_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


kmeans(img0)
kmeans(img1)
kmeans(img2)
kmeans(img3)

meansShift(img0)
meansShift(img1)
meansShift(img2)
meansShift(img3)
