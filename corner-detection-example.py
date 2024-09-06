import matplotlib.pyplot as plt
import streamlit as st

from skimage import data, transform
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.draw import ellipse

tsform = transform.AffineTransform(scale=(1.2,1.1), rotation=0.8, translation=(80,10))
image = transform.warp(data.checkerboard()[:100,:100], tsform.inverse, output_shape=(200,350))

rr, cc = ellipse(180,220,20,120)
image[rr, cc] = 1
image[40:80, 225:275] = 1
image[100:140, 275:325] = 1

coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
coords_subpix = corner_subpix(image, coords, window_size=7)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.scatter(coords[:,1], coords[:,0], color='cyan', marker='o', s=7)
ax.plot(coords_subpix[:,1], coords_subpix[:,0], '+r', markersize=16)
st.pyplot(fig)