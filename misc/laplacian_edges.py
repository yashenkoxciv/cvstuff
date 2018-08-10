import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from filters import apply, k_gaussian

k_laplacian1 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
])

k_laplacian2 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
])

def load_image(n=None):
    if n is None:
        n = np.random.randint(1, 9)
    image = (imread('{0}.jpg'.format(n), True)*255).astype(np.int)
    plt.imshow(image, cmap='gray')
    plt.show()
    return image

def preprocessing(image):
    p_image = apply(image, k_gaussian)
    plt.imshow(p_image, cmap='gray')
    plt.show()
    return p_image

def laplacian_edges(image):
    l1_img = apply(image, k_laplacian1)
    l2_img = apply(image, k_laplacian2)
    
    f, a = plt.subplots(1, 2)
    a[0].imshow(l1_img, cmap='gray')
    a[0].axis('off')
    a[1].imshow(l2_img, cmap='gray')
    a[1].axis('off')
    plt.show()
    
    return l1_img, l2_img

if __name__ == '__main__':
    image = load_image(9)
    #p_image = preprocessing(image)
    #sobel_edges(p_image)
    laplacian_edges(image)

