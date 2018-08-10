import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imshow_collection
from scipy.ndimage.filters import convolve, gaussian_filter

def canny_edges(image):
    # apply Sobel filters to get gradients (horizontal - h_g and vertical - v_g)
    h_g = convolve(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    v_g = convolve(image, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    m = np.sqrt(np.square(h_g) + np.square(v_g))
    #a = np.arctan2(v_g, h_g)
    a = (np.round(np.arctan2(v_g, h_g)*(5 / np.pi)) + 5) % 5
    
    return h_g, v_g, m, a
    

if __name__ == '__main__':
    img = imread('9.jpg', as_gray=True)
    #imshow(img)
    blur_img = gaussian_filter(img, 3.)
    o = canny_edges(blur_img)
    
    figure, axis = plt.subplots(1, 2)
    
    axis[0].imshow(img, cmap='gray')
    axis[0].axis('off')
    
    axis[1].imshow(blur_img, cmap='gray')
    axis[1].axis('off')
    
    plt.show()
    
    figure, axis = plt.subplots(1, 2)
    
    axis[0].imshow(o[0], cmap='gray')
    axis[0].axis('off')
    
    axis[1].imshow(o[1], cmap='gray')
    axis[1].axis('off')
    
    plt.show()
    
    figure, axis = plt.subplots(1, 2)
    
    axis[0].imshow(o[2], cmap='gray')
    axis[0].axis('off')
    
    axis[1].imshow(o[3], cmap='gray')
    axis[1].axis('off')
    
    plt.show()

