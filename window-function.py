import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import coffee
from skimage.filters import window

image = img_as_float(rgb2gray(coffee()))
wimage = image * window('hann', image.shape)

image_f = np.abs(fftshift(fft2(image)))
wimage_f = np.abs(fftshift(fft2(wimage)))

fig, axes = plt.subplots(2, 2)
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[1].imshow(wimage, cmap='gray')
ax[2].imshow(np.log(image_f), cmap='magma')
ax[3].imshow(np.log(wimage_f), cmap='magma')

st.pyplot(fig)
