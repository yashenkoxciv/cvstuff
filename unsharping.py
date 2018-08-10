import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters

# grayscale image
img = np.array(Image.open('imgs/3_flower.jpg').convert('L'))
gimg = filters.gaussian_filter(img, 3)
weight = 0.5
unsharp_img = img - weight*gimg

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img, cmap='gray')
axs[0].axis('off')
axs[1].imshow(gimg, cmap='gray')
axs[1].axis('off')
axs[2].imshow(unsharp_img, cmap='gray')
axs[2].axis('off')
plt.show()

# color image
img = np.array(Image.open('imgs/3_flower.jpg'), dtype=np.float) / 256
sigma = 15
gimg = np.stack([
        filters.gaussian_filter(img[:, :, 0], sigma),
        filters.gaussian_filter(img[:, :, 1], sigma),
        filters.gaussian_filter(img[:, :, 2], sigma)
], axis=2)
gimg[:, :, 0] /= gimg[:, :, 0].max() + 1 # gimg[:, :, 0].min() 
gimg[:, :, 1] /= gimg[:, :, 1].max() + 1
gimg[:, :, 2] /= gimg[:, :, 2].max() + 1
weight = .5
unsharp_mask = weight*gimg
unsharp_img = img - unsharp_mask

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img)
axs[0].axis('off')
axs[1].imshow(gimg)
axs[1].axis('off')
axs[2].imshow(unsharp_img)
axs[2].axis('off')
plt.show()

