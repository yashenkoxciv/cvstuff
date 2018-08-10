import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

def apply(image, kernel):
    if len(kernel.shape) != 2 or \
        kernel.shape[0] != kernel.shape[1] or \
        kernel.shape[0] % 2 == 0:
        raise Exception('Expected two-dimensional square kernel')
    
    k_size = kernel.shape[0]
    k_half_size = k_size // 2
    # prepend image array
    p_image = np.zeros(
            (image.shape[0]+k_half_size*2, image.shape[1]+k_half_size*2),
            np.int
    )
    p_image[k_half_size:-k_half_size, k_half_size:-k_half_size] = image
    # apply kernel
    k = np.flipud(np.fliplr(kernel))
    #k = kernel
    n_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            n_image[i, j] = np.sum(
                    p_image[i:i+k_size, j:j+k_size]*k
            )
    return n_image

k_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
])

k_gaussian = (1/273)*np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
])

k_edges = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
])

k_boxblur = (1/9)*np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
])

if __name__ == '__main__':
    img = (imread('9.jpg', True)*255).astype(np.int)
    print(img.shape, img.max(), img.min())
    p_img = apply(img, k_sharp)
    p_img = apply(p_img, k_boxblur)
    print(p_img.shape, p_img.max(), p_img.min())
    
    f, a = plt.subplots(1, 2)
    a[0].imshow(img, cmap='gray')
    a[0].axis('off')
    a[1].imshow(p_img, cmap='gray')
    a[1].axis('off')
    plt.show()
    
