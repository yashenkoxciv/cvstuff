import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from PIL import Image

def compute_harris_response(im, sigma=3):
    imx = filters.gaussian_filter(im, (sigma, sigma), (0, 1))
    imy = filters.gaussian_filter(im, (sigma, sigma), (1, 0))
    
    wxx = filters.gaussian_filter(imx*imx, sigma)
    wxy = filters.gaussian_filter(imx*imy, sigma)
    wyy = filters.gaussian_filter(imy*imy, sigma)
    
    plt.imshow(wxx)
    plt.show()
    plt.imshow(wyy)
    plt.show()
    plt.imshow(wxy)
    plt.show()
    
    wdet = wxx*wyy - wxy**2
    wtr = wxx + wyy
    return wdet / wtr

def get_harris_points(harrism, min_dist=10, threshold=0.1):
    corner_threshold = harrism.max()*threshold
    harrism_t = (harrism > corner_threshold)*1
    
    coords = np.array(harrism_t.nonzero()).T
    
    candidate_values = [harrism[c[0], c[1]] for c in coords]
    
    index = np.argsort(candidate_values)
    
    allowed_locations = np.zeros(harrism.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
                    (coords[i, 0]-min_dist):(coords[i, 0]+min_dist),
                    (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)
            ] = 0
    return filtered_coords

if __name__ == '__main__':
    img = np.uint(Image.open('imgs/9.jpg').convert('L'))
    #img = np.zeros([100, 100])
    #img[40:60, 40:60] = 255
    hr = compute_harris_response(img, 0.7)
    cc = get_harris_points(hr, threshold=0.01)
    
    print(cc)
    plt.gray()
    
    plt.imshow(img)
    plt.plot([p[1] for p in cc], [p[0] for p in cc], 'r*')
    plt.axis('off')
    plt.show()
    
    #plt.imshow(hr)
    #plt.show()


