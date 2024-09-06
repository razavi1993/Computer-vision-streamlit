import streamlit as st
import matplotlib.pyplot as plt
from skimage import data, filters
from skimage.color import rgb2gray

image = data.coffee()
edges = filters.sobel(rgb2gray(image))
fig, ax = plt.subplots()
ax.imshow(edges)
st.pyplot(fig)
