import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
# Read image 
input_image = cv.imread("test_img_dog.jpg")
new_image = []
count = 0
each_pix = 2
for y in range(len(input_image)):
    if y % 2 == 0:
        new_image.append([])
    for x in range(len(input_image[y])):
        if x % 2 == 0:
            new_image[-1].append(input_image[y][x])
new_image = np.array(new_image)

comp_image = cv.imread("test_img_dog.jpg")

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
fig.tight_layout()

# Display original image 
ax[0].imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
ax[0].set_title("Manipulated image")

#Display grayscale image
ax[1].imshow(cv.cvtColor(comp_image, cv.COLOR_BGR2RGB))
ax[1].set_title("Original image")

plt.show()
