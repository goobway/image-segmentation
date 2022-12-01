from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import io, segmentation, color
from plot_rag_merge import _weight_mean_color, merge_mean_color
from skimage.future import graph

"""
ECE597IP FINAL PROJECT 2
Author: Calista Greenway and Tyra Sofia Schoenfeldt
This program performs image segmentation on mars rover images by clustering and color space transformations.
"""

# open rover image
img_init = Image.open('images/0174.tif')

# save image as numpy array
img = np.asarray(img_init)

# implement SLIC
labels = slic(img, n_segments=300, compactness=60)
print(f'SLIC number of segments Image 2: {len(np.unique(labels))}')

# show results from ordinary SLIC
plt.figure()
plt.imshow(mark_boundaries(img, labels))
plt.title("SLIC")
plt.axis('off')

# implement Region Adjacency Graphs
g = graph.rag_mean_color(img, labels)

# show results from SLIC with RAG
plt.figure()
with_rag = graph.show_rag(labels, g, img)
plt.title("SLIC with initial RAG")
plt.axis('off')

# implement Hierarchical Merging
labels2 = graph.merge_hierarchical(labels, g, 37, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
g2 = graph.rag_mean_color(img, labels2)
 
out = color.label2rgb(labels2, img, kind='avg')
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
io.imsave('out.png',out)

