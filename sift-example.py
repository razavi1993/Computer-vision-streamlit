import matplotlib.pyplot as plt
import streamlit as st
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT

image1 = rgb2gray(data.coffee())
tsform = transform.AffineTransform(scale=(1.4,1.1), rotation=0.4, translation=(-10,-250))
image2 = transform.warp(image1, tsform)

descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(image1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(image2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)
fig, ax = plt.subplots()

plot_matches(ax, image1, image2, keypoints1, keypoints2, matches12)
st.pyplot(fig)

