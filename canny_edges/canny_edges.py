import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import convolve, gaussian_filter

'''
k_sobel_h = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
])
'''

k_sobel_h = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
])

k_sobel_v = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
])

def load_image(n=None):
    if n is None:
        n = np.random.randint(1, 11)
    image = (imread('{0}.jpg'.format(n), True)*255).astype(np.int)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return image

def preprocessing(image, k):
    p_image = gaussian_filter(image, k)
    #plt.imshow(p_image, cmap='gray')
    #plt.show()
    return p_image

def sobel_edges(image):
    seh_img = convolve(image, k_sobel_h) # horizontal gradient (approximation)
    sev_img = convolve(image, k_sobel_v) # vertical gradient
    
    f, a = plt.subplots(1, 2)
    a[0].imshow(seh_img, cmap='gray')
    a[0].axis('off')
    a[1].imshow(sev_img, cmap='gray')
    a[1].axis('off')
    plt.show()
    
    return seh_img, sev_img

def grad_props(xg, yg):
    m = np.sqrt(np.square(xg) + np.square(yg))
    d = np.arctan2(yg, xg)
    d = (np.round(d*(5. / np.pi)) + 5) % 5
    
    f, a = plt.subplots(1, 2)
    a[0].imshow(m, cmap='gray')
    a[0].axis('off')
    a[1].imshow(d, cmap='gray')
    a[1].axis('off')
    plt.show()
    
    return m, d

def nonmax_suppression(g, d, high_t=90):
    m = g.copy()
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if i == 0 or i == g.shape[0]-1 or j == 0 or j == g.shape[1]-1:
                m[i, j] = 0
                continue
            tq = d[i, j] % 4
            
            if tq == 0: # horizontal line
                if g[i, j] <= g[i, j-1] or g[i, j] <= g[i, j+1]:
                    m[i, j] = 0
            if tq == 1: # diagonal (left top corner to right bottom)
                if g[i, j] <= g[i-1, j+1] or g[i, j] <= g[i+1, j-1]:
                    m[i, j] = 0
            if tq == 2: # vertical line
                if g[i, j] <= g[i-1, j] or g[i, j] <= g[i+1, j]:
                    m[i, j] = 0
            if tq == 3: # diagonal (left bottom corner to right top)
                if g[i, j] <= g[i-1, j-1] or g[i, j] <= g[i+1, j+1]:
                    m[i, j] = 0
    plt.imshow(m, cmap='gray')
    plt.show()
    

if __name__ == '__main__':
    image = load_image(9)
    p_image = preprocessing(image, 3.)
    hg, vg = sobel_edges(p_image)
    mg, dn = grad_props(hg, vg)
    nonmax_suppression(mg, dn)
    


