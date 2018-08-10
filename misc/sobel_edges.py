import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import convolve
from filters import apply, k_gaussian

k_sobel_h = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
])

k_sobel_v = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
])

def load_image(n=None):
    if n is None:
        n = np.random.randint(1, 9)
    image = (imread('{0}.jpg'.format(n), True)*255).astype(np.int)
    plt.imshow(image, cmap='gray')
    plt.show()
    return image

def preprocessing(image):
    #p_image = apply(image, k_gaussian)
    p_image = convolve(image, k_gaussian)
    plt.imshow(p_image, cmap='gray')
    plt.show()
    return p_image

def sobel_edges(image):
    #seh_img = apply(image, k_sobel_h) # horizontal gradient (approximation)
    #sev_img = apply(image, k_sobel_v) # vertical gradient
    seh_img = convolve(image, k_sobel_h) # horizontal gradient (approximation)
    sev_img = convolve(image, k_sobel_v) # vertical gradient
    
    f, a = plt.subplots(1, 2)
    a[0].imshow(seh_img, cmap='gray')
    a[0].axis('off')
    a[1].imshow(sev_img, cmap='gray')
    a[1].axis('off')
    plt.show()
    
    return seh_img, sev_img

if __name__ == '__main__':
    image = load_image(9)
    p_image = preprocessing(image)
    sobel_edges(p_image)

