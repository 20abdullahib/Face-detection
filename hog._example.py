import matplotlib.pyplot as plt
from skimage import io, color, feature

# Load the image
image = io.imread("faces/super.jpg")

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Compute HOG features
hog_vec, hog_vis = feature.hog(gray_image, visualize=True)

# Plot the input image and visualization of HOG features
fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image)
ax[0].set_title('Input Image')

ax[1].imshow(hog_vis)
ax[1].set_title('Visualization of HOG Features')

plt.show()

# https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html