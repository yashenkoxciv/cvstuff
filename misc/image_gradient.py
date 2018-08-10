import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

def gradient(image, k=3):
    h = np.zeros_like(image)
    v = np.zeros_like(image)
    
    for i in range(image.shape[0]-k):
        for j in range(image.shape[1]-k):
            h[i, j] = np.mean(image[i, j+1:j+k+1] - image[i, j])
            v[i, j] = np.mean(image[i+1:i+k+1, j] - image[i, j])
    
    return h, v

if __name__ == '__main__':
    img = imread('9.jpg', True)
    
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    
    h, v = gradient(img, 10)
    
    f, a = plt.subplots(1, 2)
    a[0].imshow(h, cmap='gray')
    a[0].axis('off')
    a[1].imshow(v, cmap='gray')
    a[1].axis('off')
    plt.show()

