from skimage.future import graph
from skimage import segmentation, color
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

"""
ECE 597IP FINAL PROJECT 2
Author: Calista Greenway
This program performs image segmentation on mars rover images.
"""

# open rover images as numpy array (input images)
img0 = np.asarray(Image.open('images/0073.tif'))
img1 = np.asarray(Image.open('images/0174.tif'))
img2 = np.asarray(Image.open('images/0617.tif'))
img3 = np.asarray(Image.open('images/1059.tif'))

# crop images with black boarder
img0 = img0[:, 90:1512]
img2 = img2[:, 90:1536]


# felzenszwalb’s algorithm (method 1)
def felzenszwalb(input_image):
    img = input_image
    segmented_image = segmentation.felzenszwalb(
        img, scale=500, sigma=0.5, min_size=5000)
    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(segmentation.mark_boundaries(img, segmented_image, mode='thick'))
    plt.show()

    # results
    print(
        f'Felzenszwalb number of segments {len(np.unique(segmented_image))}')


# felzenszwalb over-segmentation with rag merge (method 2)
def felzenszwalb_rag(input_image):
    img = input_image
    labels = segmentation.slic(
        img, compactness=30, n_segments=400, start_label=1)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                       in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, mode='thick')
    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(out)
    plt.show()

    # results
    print(f'Felzenszwalb number of segments: {len(np.unique(out))}')


def _weight_mean_color(graph, src, dst, n):  # function from plot_rag_merge.py
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):  # function from plot_rag_merge.py
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
        graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])


# method 1
felzenszwalb(img0)
# felzenszwalb(img1)
# felzenszwalb(img2)
# felzenszwalb(img3)

# method 2
felzenszwalb_rag(img0)
# felzenszwalb_rag(img1)
# felzenszwalb_rag(img2)
# felzenszwalb_rag(img3)
